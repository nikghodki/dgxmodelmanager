"""Live container logs (GET /api/deployments/{id}/logs, admin only) are redacted line by line before they leave the
server: engine output can echo tokens, auth headers and URL credentials."""
from tests.test_api_deployments import SMALL, admin

HF = "hf_" + "Q1w2E3r4T5y6U7i8O9p0AsDf"
BEARER = "eyJhbGciOi" + "JIUzI1NiJ9xyz123abc"


def test_live_logs_are_redacted_line_by_line(api_stack):
    c = api_stack
    h = admin(c)
    dep = c.post("/dgxmm/api/deployments", headers=h, json={"name": "tiny", "engine": "vllm",
                                                             "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    cont = c.stack.sdk.container(dep["container_id"])
    cont.log_text = "\n".join([
        "INFO starting vLLM",
        f"DEBUG env HF_TOKEN={HF}",
        f"DEBUG request headers: Authorization: Bearer {BEARER}",
        "INFO cloning https://user:pa55w0rdpa55@github.com/org/repo",
        "-----BEGIN OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ", "-----END OPENSSH PRIVATE KEY-----",
        "INFO Uvicorn running on http://0.0.0.0:8100",
    ]) + "\n"
    r = c.get(f"/dgxmm/api/deployments/{dep['id']}/logs?tail=500", headers=h)
    assert r.status_code == 200
    out = r.json()["logs"]
    for secret in (HF, BEARER, "pa55w0rdpa55", "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ"):
        assert secret not in out, secret
    lines = out.splitlines()
    assert lines[0] == "INFO starting vLLM" and lines[-1] == "INFO Uvicorn running on http://0.0.0.0:8100"
    assert "HF_TOKEN=***" in out and "Authorization: ***" in out
    assert out.endswith("\n")  # line structure (and the trailing newline) is kept


def test_logs_stay_admin_only(api_stack):
    c = api_stack
    h = admin(c)
    dep = c.post("/dgxmm/api/deployments", headers=h, json={"name": "tiny", "engine": "vllm",
                                                             "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    c.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert c.get(f"/dgxmm/api/deployments/{dep['id']}/logs").status_code == 403


def test_pem_body_without_its_begin_line_is_masked():
    """`docker logs --tail` can cut the BEGIN line off: runs of base64 key-material lines are masked anyway."""
    from dgxmm.memory.redact import redact_log
    body = ["MIIEowIBAAKCAQEA" + "q" * 48, "Zm9vYmFyYmF6" * 5 + "abcd", "b3BlbnNzaC1rZXkt" * 4, "dGFpbA=="]
    text = "\n".join(["INFO x"] + body + ["-----END RSA PRIVATE KEY-----", "INFO after"]) + "\n"
    out = redact_log(text)
    assert all(b not in out for b in body[:3]), out
    assert out.startswith("INFO x\n") and out.endswith("INFO after\n")
    # ordinary log lines, hashes and a single long token-free id are left alone
    normal = "\n".join(["sha256:" + "a" * 64, "INFO Loading safetensors checkpoint shards: 100%",
                        "x" * 70, "INFO done"])
    assert redact_log(normal) == normal
