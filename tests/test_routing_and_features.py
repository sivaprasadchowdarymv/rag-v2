"""Routing (Groq text / Qwen-Gemma images), optional OpenRouter retrieval, vision OCR, cross-check, budget."""
import pymupdf

from agents.rag_pro import answer_pro
from llm.model_router import ModelRouter
from rag.pipeline import get_index, open_index
from rag.vision import make_vision
from tests.conftest import _png, control, make_datasheet, server_log


def _remote(make_settings, **kw):
    return make_settings(embed_backend="openrouter", rerank_backend="openrouter", **kw)


def test_tables_keep_spaces_and_decimals(indexed):
    index, _ = indexed
    rows = [n.raw_content for n in index.nodes if n.type == "row"]
    assert any("Output voltage | 4.8 | 5.0 | 5.2" in r for r in rows)
    assert any("Quiescent current |  | 5 | 8 mA" in r for r in rows)


def test_remote_embeddings_index_and_answer(server, sample_pdf, make_settings):
    s = _remote(make_settings, embed_dims=256)
    index, warnings = get_index(sample_pdf, "LM7805X.pdf", s)
    assert index.embed_model == "openrouter:qwen/qwen3-embedding-8b@256" and not warnings
    assert index.embeddings.shape == (len(index.nodes), 256) and index.has_embedding.all()
    log = server_log(server)
    assert log["embed_calls"] >= 1 and log["errors"] == []
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.kind == "answer" and r.verification["grounded"]
    assert server_log(server)["rerank_calls"] >= 1  # Qwen3-Reranker used for candidate ranking
    # Reopening from cache needs no new embedding calls
    before = server_log(server)["embed_calls"]
    from rag import pipeline
    pipeline._registry.clear()
    again = open_index(index.doc_id, index.filename, s)
    assert again is not None and again.has_embedding.all() and server_log(server)["embed_calls"] == before


def test_remote_rerank_failure_falls_back_to_local(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings(rerank_backend="openrouter")
    control(server, "qwen/qwen3-reranker-8b")
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.kind == "answer" and not any("Reranking was skipped" in w for w in r.warnings)


def test_embedding_outage_keeps_keyword_search(server, sample_pdf, make_settings):
    s = _remote(make_settings)
    control(server, "qwen/qwen3-embedding-8b")
    index, warnings = get_index(sample_pdf, "LM7805X.pdf", s)
    assert any("Semantic search is off" in w for w in warnings)
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.sources  # BM25 + fuzzy still find the evidence


def _scanned_pdf() -> bytes:
    doc = pymupdf.open(stream=make_datasheet(), filetype="pdf")
    page = doc.new_page()  # page 5: image only, no text layer
    page.insert_image(pymupdf.Rect(40, 40, 560, 780), stream=_png(300, 400))
    data = doc.tobytes()
    doc.close()
    return data


def test_vision_ocr_makes_scanned_pages_searchable(server, make_settings):
    s = make_settings(ocr_scanned=True, vision_index=True)
    vision = make_vision(ModelRouter(s), s)
    index, warnings = get_index(_scanned_pdf(), "scanned.pdf", s, vision=vision)
    ocr = [n for n in index.nodes if (n.meta or {}).get("ocr")]
    assert ocr and all(n.page == 5 for n in ocr) and "RthJA = 65" in " ".join(n.raw_content for n in ocr)
    assert index.doc_meta["ocr_pages"] == [5] and not index.doc_meta["scanned_pages"]
    assert not any("could not be read" in w for w in warnings)
    assert vision.calls >= 2 and index.has_embedding.all()  # OCR + figures; new nodes embedded too
    r = answer_pro("What is the thermal resistance RthJA?", [index], s, ModelRouter(s))
    assert r.kind == "answer" and any(src.page == 5 for src in r.sources)


def test_figures_get_vision_descriptions(server, make_settings, sample_pdf):
    s = make_settings(vision_index=True)
    index, _ = get_index(sample_pdf, "LM7805X.pdf", s, vision=make_vision(ModelRouter(s), s))
    described = [f for f in index.figures if (f.meta or {}).get("vision")]
    assert described and "output voltage" in described[0].meta["vision"].lower()
    r = answer_pro("Show the graph of output voltage versus input voltage", [index], s, ModelRouter(s))
    figs = [src for src in r.sources if src.type == "figure"]
    assert r.kind == "answer" and figs and all("Visual description" in f.full_text for f in figs)


def test_without_vision_scanned_pages_are_flagged(server, make_settings):
    s = make_settings()
    index, warnings = get_index(_scanned_pdf(), "scanned2.pdf", s, vision=None)
    assert index.doc_meta["scanned_pages"] == [5] and any("could not be read" in w for w in warnings)


def test_llm_cross_check_uses_groq_llama(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings(verify_with_llm=True)
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.verification["llm_checked"] and r.verifier_model == "llama-3.3-70b-versatile"
    assert r.verification["verdict"] in ("verified", "partial")


def test_llm_rejection_triggers_one_correction(server, indexed, make_settings):
    index, _ = indexed
    import urllib.request
    urllib.request.urlopen(server + "/_ctl?verifier=reject")
    s = make_settings(verify_with_llm=True)
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.regenerated and r.llm_calls == 2


def test_master_fallback_is_visible(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings()
    control(server, "groq:openai/gpt-oss-120b")
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert r.model == "llama-3.3-70b-versatile"
    assert any("Model fallback used" in st.text for st in r.steps)


def test_daily_budget_stops_ai_answers(server, indexed, make_settings):
    from ops import telemetry
    index, _ = indexed
    s = make_settings(daily_budget_usd=0.01)
    telemetry.record(s.data_dir, question="q", latency_s=1, tokens=1, llm_calls=1, provider="openrouter",
                     cached=False, error=False, kind="answer", cost_usd=0.02)
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    assert not r.used_llm and r.sources and any("budget" in w for w in r.warnings)


def test_text_question_uses_groq_only(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings()
    r = answer_pro("What is the peak output current?", [index], s, ModelRouter(s))
    reqs = server_log(server)["requests"]
    assert r.provider == "groq" and r.cost == 0.0
    assert all(kind != "chat" for kind, _ in reqs)  # no OpenRouter chat call for a text question


def test_figure_question_sends_images_to_gemma(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings()
    r = answer_pro("What does the graph of output voltage vs input voltage show?", [index], s, ModelRouter(s))
    assert r.provider == "openrouter" and r.model == "google/gemma-4-31b-it:free" and r.llm_calls == 1
    assert any("image(s) sent to the vision model" in st.text for st in r.steps)
    assert any(src.type == "figure" for src in r.sources)
    # 5.1 and 7 were read from the image: accepted, but honestly marked "partly verified"
    assert r.verification["grounded"] and r.verification["verdict"] == "partial"
    assert any("read from figure images" in n for n in r.verification["notes"])
    assert server_log(server)["errors"] == []  # no image ever reached Groq


def test_figure_question_without_openrouter_uses_captions(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings(openrouter_api_key="")
    r = answer_pro("What does the graph of output voltage vs input voltage show?", [index], s, ModelRouter(s))
    assert r.provider == "groq" and r.kind == "answer"


def test_vision_outage_falls_back_to_text_model(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings()
    control(server, "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,qwen/qwen3.8-27b")
    r = answer_pro("What does the graph of output voltage vs input voltage show?", [index], s, ModelRouter(s))
    assert r.provider == "groq" and any("figure captions" in w for w in r.warnings)


def test_image_answers_can_be_switched_off(server, indexed, make_settings):
    index, _ = indexed
    s = make_settings(image_answers=False, vision_models=("openrouter:google/gemma-4-31b-it:free",))
    r = answer_pro("What does the graph of output voltage vs input voltage show?", [index], s, ModelRouter(s))
    assert r.provider == "groq"


def test_retrieval_is_local_by_default_and_auto_follows_profile(make_settings):
    from config.settings import Settings
    from rag.embeddings import active_model
    assert Settings().embed_backend == "local" and Settings().rerank_backend == "local"  # 0 tokens by default
    paid = make_settings(embed_backend="auto", rerank_backend="auto", model_profile="quality")
    free = make_settings(embed_backend="auto", rerank_backend="auto", model_profile="free")
    assert active_model(paid).startswith("openrouter:qwen/qwen3-embedding-8b") and paid.remote_rerank
    assert not free.remote_embeddings and not free.remote_rerank  # $0 profile: local retrieval models
    assert not make_settings(embed_backend="auto", openrouter_api_key="").remote_embeddings
