"""Command-line redaction for the processes view (spec §5). Secrets must never reach the client."""
from dgxmm.memory.redact import MAX_COMMAND_BYTES, format_command, redact_argv

HF = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
SK = "sk-" + "proj-4f9aZ_x8Qw2LmN0pR7tUvW"
SECRETS = ["s3cr3t-api-key-value", "hunter2", "pgpass", HF, SK, "eyJhbGciOiJIUzI1NiJ9.payload.sig", "tok123456789"]


def cmd(argv):
    out = format_command(argv)[0]
    for s in SECRETS:
        assert s not in out, f"{s!r} leaked in {out!r}"
    return out


def test_vllm_serve_api_key_space_and_equals():
    assert cmd(["vllm", "serve", "Qwen/Qwen3-8B", "--api-key", "s3cr3t-api-key-value", "--port", "8000"]) == \
        "vllm serve Qwen/Qwen3-8B --api-key *** --port 8000"
    assert cmd(["vllm", "serve", "m", "--api-key=s3cr3t-api-key-value"]) == "vllm serve m --api-key=***"


def test_vllm_non_secret_flags_are_kept_verbatim():
    argv = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/Qwen3-8B", "--tokenizer",
            "Qwen/Qwen3-8B", "--tokenizer-mode", "auto", "--max-num-batched-tokens", "8192", "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.35", "--enable-auto-tool-choice", "--served-model-name", "qwen3-8b"]
    assert format_command(argv)[0] == " ".join(argv)


def test_litellm_config_with_database_url_userinfo():
    out = cmd(["/bin/sh", "-c", "DATABASE_URL=postgres://llm:pgpass@litellm-postgres:5432/litellm litellm --config /app/config.yaml --port 4000"])
    assert "postgres://***@litellm-postgres:5432/litellm" in out
    assert "--config /app/config.yaml --port 4000" in out


def test_key_value_tokens_whose_key_matches():
    out = cmd(["env", "HF_TOKEN=" + HF, "LITELLM_MASTER_KEY=hunter2", "OPENAI_API_KEY=x" + "y" * 20, "PORT=4000", "python", "x.py"])
    assert out.startswith("env HF_TOKEN=*** LITELLM_MASTER_KEY=*** OPENAI_API_KEY=*** PORT=4000 python x.py")


def test_python_dash_c_with_inline_tokens():
    out = cmd(["python", "-c", f"import os; key='{SK}'; tok=\"{HF}\"; client(api_key='hunter2', token='tok123456789')"])
    assert "api_key=" in out and "token=" in out and "import os" in out


def test_hf_and_sk_tokens_anywhere():
    out = cmd(["huggingface-cli", "download", "org/m", HF])
    assert out == "huggingface-cli download org/m ***"
    assert cmd(["curl", f"https://api.example.com/?k={SK}"]) == "curl 'https://api.example.com/?k=***'"


def test_password_flags_and_short_flags():
    assert cmd(["mysql", "--password=hunter2", "-u", "root"]) == "mysql --password=*** -u root"
    assert cmd(["redis-cli", "--pass", "hunter2", "-p", "6379"]) == "redis-cli --pass *** -p 6379"
    assert cmd(["svc", "--auth-token", "tok123456789"]) == "svc --auth-token ***"
    assert cmd(["svc", "--client-secret", "hunter2", "--credentials-file", "/etc/c.json"]) == \
        "svc --client-secret *** --credentials-file ***"
    # a sensitive boolean flag followed by another flag does not swallow it
    assert cmd(["svc", "--disable-auth", "--port", "1"]) == "svc --disable-auth --port 1"


def test_bearer_headers():
    out = cmd(["curl", "-H", "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "http://x/v1/models"])
    assert "Bearer ***" in out and "http://x/v1/models" in out
    assert "bearer ***" in cmd(["sh", "-c", "curl -H 'authorization: bearer tok123456789' http://x"])


def test_flags_inside_shell_strings():
    out = cmd(["bash", "-c", "exec vllm serve m --api-key s3cr3t-api-key-value --port 8000"])
    assert "--api-key *** --port 8000" in out
    out = cmd(["sh", "-c", "llama-server -m /m.gguf --api-key=hunter2"])
    assert "--api-key=***" in out


def test_url_userinfo_and_query_tokens():
    out = cmd(["redis-server", "--replicaof", "redis://default:hunter2@redis:6379/0"])
    assert "redis://***@redis:6379/0" in out
    out = cmd(["wget", "https://h/x?token=tok123456789&x=1"])
    assert "token=***" in out and "x=1" in out


def test_json_style_pairs():
    out = cmd(["app", "--hf-overrides", '{"api_key": "hunter2", "rope": 2}'])
    assert '"rope": 2' in out


def test_shell_quoting_and_truncation():
    assert format_command(["echo", "a b", "it's"])[0] == "echo 'a b' 'it'\"'\"'s'"
    long_argv = ["python", "-c", "x" * 10_000]
    out, truncated = format_command(long_argv)
    assert truncated and len(out.encode()) <= MAX_COMMAND_BYTES + 3
    assert format_command(["a"]) == ("a", False)


def test_secret_split_by_truncation_boundary_is_still_redacted():
    # the key is redacted before truncation, so no partial secret survives the cut
    argv = ["x" * (MAX_COMMAND_BYTES - 12), "--api-key", "s3cr3t-api-key-value"]
    out, truncated = format_command(argv)
    assert truncated and "s3cr3t" not in out


def test_empty():
    assert format_command([]) == ("", False)


def test_redact_argv_keeps_argument_count():
    argv = ["vllm", "--api-key", "s3cr3t-api-key-value"]
    assert redact_argv(argv) == ["vllm", "--api-key", "***"]


def test_sensitive_name_rules():
    from dgxmm.memory.redact import is_sensitive_name
    for name in ("api-key", "api_key", "apikey", "hf-token", "HF_TOKEN", "use-auth-token", "authorization", "secret",
                 "client-secret", "password", "PGPASSWORD", "passwd", "pass", "credentials", "LITELLM_MASTER_KEY",
                 "tokenizer-secret", "ssl-keyfile"):
        assert is_sensitive_name(name), name
    for name in ("model", "port", "tokenizer", "tokenizer-mode", "max-tokens", "max-num-batched-tokens",
                 "enable-auto-tool-choice", "served-model-name", "passthrough", "host", "config"):
        assert not is_sensitive_name(name), name


def test_trailing_sensitive_flag_and_negative_number_value():
    assert cmd(["svc", "--api-key"]) == "svc --api-key"
    assert cmd(["svc", "--token", "-123"]) == "svc --token ***"
