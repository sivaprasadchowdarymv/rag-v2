"""2.1 metadata intelligence: extraction with provenance/confidence, unit normalisation, user edits, filtering."""
from __future__ import annotations

import pytest

from rag.metadata_intel import build_profile, effective, matches, normalize_unit
from storage.metadata_store import MetadataError, MetadataStore, resolve_doc_filter, sanitize_user_metadata, scope_for

PW = "Str0ngPassw0rd!"


@pytest.mark.parametrize("value,unit,expect", [
    (500, "mA", (0.5, "A", "current")), (5.0, "V", (5.0, "V", "voltage")), (10, "kΩ", (10000.0, "Ω", "resistance")),
    (100, "nF", (1e-07, "F", "capacitance")), (2, "MHz", (2e6, "Hz", "frequency")), (212, "°F", (100.0, "°C", "temperature")),
    (3, "mm", (0.003, "m", "length")), (7, "furlong", (None, "furlong", "")),
])
def test_unit_normalisation(value, unit, expect):
    si, u, kind = normalize_unit(value, unit)
    assert (si if si is None else pytest.approx(si)) == expect[0] and u == expect[1] and kind == expect[2]


def test_profile_of_datasheet(indexed):
    index, _ = indexed
    prof = build_profile(index, {"size_bytes": 1234, "sensitivity": "internal"})
    doc = prof["document"]
    assert doc["title"]["value"].startswith("LM7805X") and doc["title"]["source"] == "file"
    assert doc["pages"]["value"] == 4 and doc["file_size_bytes"]["value"] == 1234
    assert doc["category"]["value"] == "datasheet" and doc["category"]["confidence"] >= 0.6
    by = {(p["name"], p["qualifier"], p["unit"]): p for p in prof["parameters"]}
    vout = by[("VOUT", "", "V")]
    assert vout["si_value"] == 5.0 and vout["kind"] == "voltage" and 1 in vout["pages"] and vout["source"] == "auto"
    io = by[("IO", "", "mA")]
    assert io["si_value"] == pytest.approx(0.5) and io["si_unit"] == "A"  # original "500mA" kept in raw
    assert "500mA" in io["raw"].replace(" ", "")
    peak = [p for p in prof["parameters"] if p["name"] == "Peak output current"]
    assert peak and peak[0]["unit"] == "A" and peak[0]["origin"] == "table" and peak[0]["confidence"] < vout["confidence"]
    assert any(p["value"] == "LM7805X" for p in prof["part_numbers"])
    assert len(prof["references"]["figures"]) == 2 and prof["references"]["tables"]
    assert prof["counts"]["equations"] > 0 and "datasheet" in prof["suggested_tags"]


def test_parameters_are_deduplicated(indexed):
    index, _ = indexed
    params = build_profile(index)["parameters"]
    keys = [(p["name"].lower(), p["qualifier"], p["si_value"], p["si_unit"]) for p in params]
    assert len(keys) == len(set(keys))
    vin10 = [p for p in params if p["name"] == "VIN" and p["value"] == 10]
    assert vin10 and vin10[0]["occurrences"] >= 2  # appears in text and in the equation node


def test_user_values_win_and_are_marked(indexed):
    index, _ = indexed
    prof = build_profile(index)
    eff = effective(prof, {"title": "My regulator", "tags": ["power", "5v"]})
    assert eff["title"] == {"value": "My regulator", "source": "user", "confidence": 1.0}
    assert eff["tags"]["value"] == ["power", "5v"] and eff["author"]["source"] in ("file", "auto")
    assert matches(prof, {"tags": ["power"]}, tags=["power"]) and not matches(prof, {}, tags=["power"])
    assert matches(prof, {}, part_number="lm7805") and matches(prof, {}, query="regulator vout")
    assert not matches(prof, {}, category="manual")


def test_sanitisation():
    clean = sanitize_user_metadata({"title": "  A\x00 title  ", "tags": "Power, power , 5V", "part_numbers": ["lm7805"],
                                    "custom": {"Supplier": "ACME" * 200}})
    assert clean["title"] == "A title" and clean["tags"] == ["power", "5v"] and clean["part_numbers"] == ["LM7805"]
    assert len(clean["custom"]["Supplier"]) == 300
    with pytest.raises(MetadataError):
        sanitize_user_metadata({"owner": "me"})
    with pytest.raises(MetadataError):
        sanitize_user_metadata({"custom": ["x"]})


def test_store_scopes_are_isolated_and_persist(indexed):
    index, s = indexed
    store = MetadataStore(s)
    store.update_user(index.doc_id, "tenant:acme", {"category": "Manual", "tags": ["pumps"]}, actor="u1")
    assert store.user(index.doc_id, "tenant:acme")["category"] == "manual"
    assert store.user(index.doc_id, "tenant:globex") == {}
    v = store.view(index, "tenant:acme")
    assert v["effective"]["category"]["source"] == "user" and v["profile"]["profile_version"] >= 1
    store.update_user(index.doc_id, "tenant:acme", {"category": ""})  # empty → back to the suggestion
    assert store.view(index, "tenant:acme")["effective"]["category"]["value"] == "datasheet"
    store.reset_user(index.doc_id, "tenant:acme")
    assert store.user(index.doc_id, "tenant:acme") == {}
    # profile is cached on disk and reused
    assert (s.data_dir / "documents" / index.doc_id / "metadata_profile.json").exists()


def test_scope_for():
    from core.service import Principal

    assert scope_for(Principal(tenant_id="acme"), "wsX", "users") == "tenant:acme"
    assert scope_for(Principal(), "wsX", "none") == "ws:wsX"


def test_resolve_doc_filter(indexed):
    index, s = indexed
    assert resolve_doc_filter(None, [index], s, "x") is None
    assert resolve_doc_filter({"types": ["table"]}, [index], s, "x") == {"types": ["table"]}
    hit = resolve_doc_filter({"category": "datasheet", "types": ["table"]}, [index], s, "x")
    assert hit == {"types": ["table"], "doc_ids": [index.doc_id]}
    miss = resolve_doc_filter({"tags": "pumps"}, [index], s, "x")
    assert miss["doc_ids"] == ["__no_document_matches__"]


def test_node_level_metadata_filter(indexed):
    from retrieval.hybrid_search import HybridSearcher, MetadataFilter

    index, s = indexed
    flt = MetadataFilter.from_dict({"parameter_kinds": ["temperature"]})
    hits, _ = HybridSearcher(s).search("operating temperature", [index], k=5, mode="hybrid", filters=flt)
    assert hits and all("temperature" in (h.node.meta or {}) for h in hits)
    assert MetadataFilter.from_dict({"part_numbers": "lm7805x"}).part_numbers == {"LM7805X"}
    assert MetadataFilter.from_dict({"types": ["table"]}).part_numbers is None  # old filters unchanged


def test_service_applies_document_metadata_filter(make_settings, server, sample_pdf):
    from core.service import AskRequest, RAGService
    from ingestion.jobs import IngestionService
    from security.auth import AuthService

    s = make_settings(auth_mode="users", rate_limit_per_min=0, global_rate_limit_per_min=0)
    auth = AuthService(s)
    auth.create_user("m_admin", PW, ["admin"], "acme")
    admin = auth.authenticate("m_admin", PW)
    res = IngestionService(s).ingest(sample_pdf, "LM7805X.pdf", admin)
    svc = RAGService(s)
    q = "What is the maximum output current?"
    ok = svc.ask(AskRequest(q, [res.index], metadata_filter={"category": "datasheet"}), admin)
    assert ok.kind == "answer" and ok.sources
    none = svc.ask(AskRequest(q, [res.index], metadata_filter={"tags": ["no-such-tag"]}), admin)
    assert not none.sources


def test_metadata_api(make_settings, server, sample_pdf):
    from fastapi.testclient import TestClient

    from api.main import create_app
    from ingestion.jobs import IngestionService
    from security.auth import AuthService

    s = make_settings(auth_mode="users", rate_limit_per_min=0, global_rate_limit_per_min=0)
    auth = AuthService(s)
    auth.create_user("api_admin", PW, ["admin"], "acme")
    auth.create_user("api_viewer", PW, ["viewer"], "acme")
    auth.create_user("other_admin", PW, ["admin"], "globex")
    res = IngestionService(s).ingest(sample_pdf, "LM7805X.pdf", auth.authenticate("api_admin", PW))
    client = TestClient(create_app(s))

    def hdr(user):
        return {"Authorization": "Bearer " + client.post("/v1/auth/token",
                                                         json={"username": user, "password": PW}).json()["token"]}

    a, v, o = hdr("api_admin"), hdr("api_viewer"), hdr("other_admin")
    got = client.get(f"/v1/documents/{res.document_id}/metadata", headers=v).json()
    assert got["effective"]["category"]["value"] == "datasheet" and got["profile"]["parameters"]
    assert client.patch(f"/v1/documents/{res.document_id}/metadata", json={"tags": ["x"]}, headers=v).status_code == 403
    r = client.patch(f"/v1/documents/{res.document_id}/metadata", json={"tags": ["Power"], "category": "Reference"},
                     headers=a)
    assert r.status_code == 200 and r.json()["effective"]["tags"]["value"] == ["power"]
    assert client.patch(f"/v1/documents/{res.document_id}/metadata", json={"bogus": 1}, headers=a).status_code == 422
    assert [d["id"] for d in client.get("/v1/metadata/search?tag=power", headers=v).json()] == [res.document_id]
    assert client.get("/v1/metadata/search?tag=power", headers=o).json() == []                 # other tenant
    assert client.get(f"/v1/documents/{res.document_id}/metadata", headers=o).status_code == 404
