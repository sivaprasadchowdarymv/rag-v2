import os
from streamlit.testing.v1 import AppTest
from tests.integration.test_ui_upgrade import _reset_process_caches

def test_models_page_key_table(tmp_path, fake_llm):
    env = {"DATA_DIR": str(tmp_path), "AUTH_MODE": "users", "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "Adm1nPassw0rd",
           "GROQ_API_KEY": "gsk_test", "GROQ_API_KEY_2": "gsk_backup_SECRET", "GROQ_BASE_URL": fake_llm + "/openai/v1",
           "OPENROUTER_API_KEY": "sk-or-test", "OPENROUTER_BASE_URL": fake_llm + "/api/v1"}
    old = {k: os.environ.get(k) for k in env}; os.environ.update(env); _reset_process_caches()
    try:
        from config.settings import load_settings
        from security.auth import AuthService
        AuthService(load_settings()).bootstrap_admin()
        at = AppTest.from_file("../ui_harness.py", default_timeout=90); at.query_params["page"] = "models"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        tables = [df.value for df in at.dataframe if "key" in df.value.columns]
        assert tables and list(tables[0]["key"]) == ["key #1", "key #2", "key #1"]
        assert "SECRET" not in tables[0].to_string()
    finally:
        _reset_process_caches()
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
