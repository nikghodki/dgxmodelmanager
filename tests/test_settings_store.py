from dgxmm.db import AppSetting, Database
from dgxmm.settings_store import SettingsStore


def test_get_set_delete_and_cache(settings, db):
    st = SettingsStore(db, settings)
    assert st.get("x") is None and st.get("x", "d") == "d"
    st.set("x", "1")
    assert st.get("x") == "1"
    with db.session() as s:
        row = s.get(AppSetting, "x")
        assert row.value == "1" and row.updated_at is not None
    # a second store over the same DB sees the persisted value
    assert SettingsStore(Database(f"sqlite:///{settings.db_path}"), settings).get("x") == "1"
    st.set("x", None)
    assert st.get("x") is None
    with db.session() as s:
        assert s.get(AppSetting, "x") is None


def test_typed_bool(settings, db):
    st = SettingsStore(db, settings)
    assert st.get_bool("b", True) is True
    st.set_bool("b", False)
    assert st.get("b") == "false" and st.get_bool("b", True) is False
    st.set("b", "garbage")
    assert st.get_bool("b", True) is True


def test_litellm_defaults_from_config(settings, db, tmp_path):
    st = SettingsStore(db, settings)
    c = st.litellm()
    assert (c.enabled, c.url, c.public_url, c.master_key, c.master_key_source) == \
        (False, settings.litellm_url, "", None, "none")
    (tmp_path / "litellm_key").write_text("sk-from-file\n")
    c = st.litellm()
    assert (c.enabled, c.master_key, c.master_key_source) == (True, "sk-from-file", "secret_file")
    assert "sk-from-file" not in repr(c)
    assert c.endpoint == settings.litellm_url + "/v1"


def test_litellm_settings_take_precedence(settings, db, tmp_path):
    (tmp_path / "litellm_key").write_text("sk-from-file")
    st = SettingsStore(db, settings)
    st.set("litellm.url", "http://other:4000/")
    st.set("litellm.public_url", "https://llm.example.com/")
    st.set_master_key("sk-from-db", "http://other:4000")
    st.set_bool("litellm.enabled", False)
    c = st.litellm()
    assert (c.enabled, c.url, c.public_url, c.master_key, c.master_key_source) == \
        (False, "http://other:4000", "https://llm.example.com", "sk-from-db", "settings")
    assert c.endpoint == "https://llm.example.com/v1"
    st.set_master_key(None)
    assert st.litellm().master_key_source == "none"  # the file key belongs to the default URL only
    st.set("litellm.url", None)
    assert st.litellm().master_key_source == "secret_file"


def test_app_context_has_store(ctx):
    assert isinstance(ctx.settings_store, SettingsStore)


def test_file_key_only_for_default_url(settings, db, tmp_path):
    (tmp_path / "litellm_key").write_text("sk-from-file")
    st = SettingsStore(db, settings)
    st.set("litellm.url", "http://elsewhere:4000")
    c = st.litellm()
    assert (c.master_key, c.master_key_source) == (None, "none")
    assert c.master_key_hint and "re-enter" in c.master_key_hint
    st.set("litellm.url", settings.litellm_url + "/")  # same URL, trailing slash
    assert st.litellm().master_key == "sk-from-file"


def test_stored_key_bound_to_its_url(settings, db):
    st = SettingsStore(db, settings)
    st.set_master_key("sk-db", "http://a:4000/")
    st.set("litellm.url", "http://a:4000")
    assert (st.litellm().master_key, st.litellm().master_key_source) == ("sk-db", "settings")
    st.set("litellm.url", "http://b:4000")
    c = st.litellm()
    assert c.master_key is None and c.master_key_source == "none" and "re-enter" in c.master_key_hint
    st.set_master_key(None)
    assert st.get("litellm.master_key") is None and st.get("litellm.master_key_url") is None
