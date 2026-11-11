"""Phase 2: multi-format ingestion, validation, chunking strategies, versioning, background jobs."""
from __future__ import annotations


import pytest

from core.service import Principal
from ingestion import ocr
from ingestion.chunker import chunk_page
from ingestion.jobs import IngestionService
from ingestion.processor import parse_document
from ingestion.validation import IngestionError, validate_upload
from rag.pipeline import open_index
from tests import fixtures_docs as fx

P = Principal("u1", "t1", ("editor",))


@pytest.mark.parametrize("name,data,expect", [
    ("manual.docx", fx.make_docx, ["24 V DC", "Standby power | 1.5 | W"]),
    ("deck.pptx", fx.make_pptx, ["Revenue grew 12 percent", "South | 1.7", "missed target"]),
    ("stock.xlsx", fx.make_xlsx, ["Bearing 6204 | 140 | Aisle 3", "Bearing 6204 | 4.75"]),
    ("limits.csv", lambda: fx.CSV.encode(), ["pressure | 6.5 | bar"]),
    ("guide.md", lambda: fx.MARKDOWN.encode(), ["port 8443"]),
    ("warranty.html", lambda: fx.HTML.encode(), ["36 months", "Motor | 24"]),
    ("device.json", lambda: fx.JSON.encode(), ["device.limits.max_temp_c: 70"]),
])
def test_every_format_is_indexed(name, data, expect, make_settings, tmp_path):
    parsed = parse_document(data(), name, make_settings(), tmp_path / "f")
    blob = "\n".join(n.raw_content for n in parsed.nodes)
    for e in expect:
        assert e in blob, (name, e)
    assert parsed.doc_meta["format"] == name.rsplit(".", 1)[1]


def test_html_scripts_are_dropped(make_settings, tmp_path):
    parsed = parse_document(fx.HTML.encode(), "w.html", make_settings(), tmp_path / "f")
    assert "alert" not in " ".join(n.raw_content for n in parsed.nodes)
    assert parsed.doc_meta["title"] == "Warranty"


def test_markdown_sections_and_tables_become_nodes(make_settings, tmp_path):
    parsed = parse_document(fx.make_docx(), "m.docx", make_settings(), tmp_path / "f")
    assert {"Electrical Ratings", "Maintenance"} <= {s for s, _ in parsed.doc_meta["sections"]}
    assert any(n.type == "table" for n in parsed.nodes) and any(n.type == "row" for n in parsed.nodes)


@pytest.mark.skipif(not ocr.available(), reason="tesseract not installed")
def test_image_ocr(make_settings, tmp_path):
    parsed = parse_document(fx.make_png_text(), "scan.png", make_settings(), tmp_path / "f")
    text = " ".join(n.raw_content for n in parsed.nodes).upper()
    assert "INVOICE" in text and "1250" in text
    assert parsed.figures and parsed.figures[0].figure_file and parsed.doc_meta["ocr_pages"] == [1]


_BAD = {
    "exe-as-pdf": ("evil.pdf", lambda: b"MZ\x90\x00" + b"\x00" * 100, "Executable"),
    "pdf-as-docx": ("fake.docx", lambda: b"%PDF-1.7 not a docx", "not a valid .docx"),
    "macro-docx": ("macro.docx", fx.make_macro_docx, "Macro"),
    "zip-bomb": ("bomb.docx", fx.make_zip_bomb_docx, "zip bomb"),
    "bad-ext": ("x.exe", lambda: b"hello", "not accepted"),
    "html-as-png": ("pic.png", lambda: b"<html>not an image</html>", "not a valid PNG"),
    "bad-json": ("bad.json", lambda: b"{not json", "not valid JSON"),
    "binary-txt": ("bin.txt", lambda: b"abc\x00\x01\x02" * 50, "binary"),
    "empty": ("empty.csv", lambda: b"", "empty"),
}


@pytest.mark.parametrize("case", list(_BAD), ids=list(_BAD))
def test_malicious_or_invalid_files_are_rejected(case, make_settings):
    name, data, msg = _BAD[case]
    with pytest.raises(IngestionError) as exc:
        validate_upload(data(), name, make_settings())
    assert msg.lower() in exc.value.user_message.lower()


def test_size_limit(make_settings):
    with pytest.raises(IngestionError, match="limit"):
        validate_upload(b"a" * (2 * 1024 * 1024), "big.txt", make_settings(max_upload_mb=1))


@pytest.mark.parametrize("strategy", ["recursive", "fixed", "semantic", "structure"])
def test_chunk_strategies_respect_size(strategy, make_settings):
    s = make_settings(chunk_strategy=strategy, max_chunk_tokens=100, overlap_tokens=10)
    text = "# Intro\n\n" + " ".join(f"Sentence number {i} talks about pumps and valves." for i in range(80))
    nodes = chunk_page(text, 1, s).nodes
    assert len(nodes) >= 3
    assert all(len(n.raw_content) <= 100 * 4 * 1.6 for n in nodes), strategy
    assert all(n.type == "text" and n.page == 1 for n in nodes)


def test_chunk_strategy_changes_cache_key(make_settings):
    from storage.vector_store import parse_key
    assert parse_key(make_settings()) != parse_key(make_settings(chunk_strategy="fixed"))


def test_ingest_versioning_dedup_and_delete(make_settings, server):
    s = make_settings()
    svc = IngestionService(s)
    r1 = svc.ingest(fx.MARKDOWN.encode(), "guide.md", P)
    assert r1.status == "new" and r1.version == 1
    assert svc.ingest(fx.MARKDOWN.encode(), "copy.md", P).status == "duplicate"
    r2 = svc.ingest((fx.MARKDOWN + "\nVersion two adds port 9000.\n").encode(), "guide.md", P)
    assert r2.status == "new_version" and r2.version == 2 and r2.document_id == r1.document_id
    assert r1.index.doc_id in r2.superseded_hashes
    assert open_index(r1.index.doc_id, "guide.md", s) is None  # old vectors removed: nothing stale
    from security.policy_engine import PolicyEngine
    pol = PolicyEngine(s, svc.repo)
    assert not pol.can_read(P, r1.index.doc_id) and pol.can_read(P, r2.index.doc_id)
    with pytest.raises(PermissionError):
        svc.delete(r2.document_id, Principal("u2", "t1", ("viewer",)))
    orphans = svc.delete(r2.document_id, P)
    assert r2.index.doc_id in orphans and open_index(r2.index.doc_id, "guide.md", s) is None
    assert svc.store.list("t1/") == []  # originals of all versions removed


def test_pii_document_is_confidential(make_settings, server):
    svc = IngestionService(make_settings())
    r = svc.ingest(b"Employee record\nPAN: ABCPE1234F\nAccount number: 123456789012\n", "hr.txt", P)
    assert r.pii_detected and r.sensitivity == "confidential"


def test_background_job_success_failure_and_retry(make_settings, server, monkeypatch):
    s = make_settings(ingest_retries=2)
    svc = IngestionService(s)
    ok = svc.submit(fx.CSV.encode(), "limits.csv", P)
    assert svc.wait(ok).status == "new"
    job = svc.repo.get_job(ok)
    assert job.status == "succeeded" and job.progress == 1.0 and job.attempts == 1
    bad = svc.submit(b"MZ" + b"\x00" * 10, "x.pdf", P)
    svc.wait(bad)
    job = svc.repo.get_job(bad)
    assert job.status == "failed" and "Executable" in job.error and job.attempts == 0
    import ingestion.jobs as jobs
    real, calls = jobs.get_index, {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return real(*a, **kw)
    monkeypatch.setattr(jobs, "get_index", flaky)
    rid = svc.submit(fx.JSON.encode(), "device.json", P)
    assert svc.wait(rid).status == "new" and svc.repo.get_job(rid).attempts == 2
    assert [j.status for j in svc.repo.list_jobs("t1", status="failed")] == ["failed"]
