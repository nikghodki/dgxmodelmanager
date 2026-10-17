"""The failed-deployment log tail keeps root-cause lines that scrolled out of the last 200 lines (real case:
`ld: final link failed: No space left on device` 487 lines above the end of a vLLM start-up log)."""
from dgxmm.deployer.logtail import TAIL_LINES, condense
from tests.test_deployer import req


def test_short_log_is_unchanged():
    text = "\n".join(f"line {i}" for i in range(TAIL_LINES))
    assert condense(text) == text


def test_long_log_keeps_earlier_error_lines_and_the_tail():
    lines = [f"INFO step {i}" for i in range(520)]
    lines[32] = "/usr/bin/ld: final link failed: No space left on device"
    lines[33] = "collect2: error: ld returned 1 exit status"
    lines[40] = "  File \"/x.py\", line 3, in f"  # traceback frames are not error lines
    out = condense("\n".join(lines))
    assert "No space left on device" in out and "collect2: error" in out
    assert out.splitlines()[-TAIL_LINES:] == lines[-TAIL_LINES:]
    assert "320 earlier lines" in out and 'File "/x.py"' not in out
    assert len(out.splitlines()) <= TAIL_LINES + 2 + 40


def test_earlier_error_lines_are_capped_and_deduplicated():
    lines = [f"ERROR retry {i % 3}" for i in range(1000)] + ["INFO ok"] * TAIL_LINES
    out = condense("\n".join(lines))
    assert out.count("ERROR retry 1") == 1
    assert len(out.splitlines()) <= TAIL_LINES + 2 + 40


async def test_failed_deployment_stores_condensed_log(stack):
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text = "\n".join(["/usr/bin/ld: final link failed: No space left on device"] +
                           [f"(EngineCore) frame {i}" for i in range(400)])
    c.status, c.exit_code = "exited", 1
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    tail = stack.deployer.view(dep["id"], with_events=True)["events"][0]["log_tail"]
    assert "No space left on device" in tail and "frame 399" in tail


def test_vllm_error_level_traceback_dump_is_not_error_lines():
    """Real vLLM output: tracebacks re-logged at ERROR level must not crowd out the real root cause."""
    lines = ["/usr/bin/ld: final link failed: No space left on device"]
    lines += ['(EngineCore pid=148) ERROR 10-17 01:50:53 [core.py:1368]   File "/x/simd.py", line 4510, in gen',
              "(EngineCore pid=148) ERROR 10-17 01:50:53 [core.py:1368]     src_code = self.generate(nodes)",
              "(EngineCore pid=148) ERROR 10-17 01:50:53 [core.py:1368]                ^^^^^^^^^^^^^^^^^^^"] * 60
    lines += ["(EngineCore pid=148) ERROR 10-17 01:50:53 [core.py:1368] torch._inductor.exc.InductorError: boom"]
    lines += ["INFO tail"] * TAIL_LINES
    out = condense("\n".join(lines))
    assert "No space left on device" in out and "InductorError: boom" in out
    assert "src_code" not in out.split("[last")[0]


HF_TOKEN = "hf_" + "Q1w2E3r4T5y6U7i8O9p0AsDfGh"
BEARER = "Bearer " + "abcDEF123ghiJKL456mnoPQR"


def _secret_log():
    return "\n".join([f"(APIServer pid=1) using token {HF_TOKEN}", f"Authorization: {BEARER}",
                      "/usr/bin/ld: final link failed: No space left on device"] +
                     [f"(EngineCore) frame {i}" for i in range(400)] + [f"retrying with {HF_TOKEN}"])


async def test_stored_log_tail_is_redacted(stack):
    """Secrets far above the stored tail (and inside it) never reach the deployment event."""
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text, c.status, c.exit_code = _secret_log(), "exited", 1
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    from dgxmm.db import DeploymentEvent
    with stack.db.session() as s:
        stored = [e.log_tail for e in s.query(DeploymentEvent).filter_by(deployment_id=dep["id"]) if e.log_tail]
    assert stored and all(HF_TOKEN not in t and "abcDEF123ghiJKL456mnoPQR" not in t for t in stored)
    assert "No space left on device" in stored[-1]


def test_api_never_returns_unredacted_log_tails(api_stack):
    """Rows written before redaction-at-store (older versions) are redacted when they are read."""
    from dgxmm.db import Deployment, DeploymentEvent
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    with api_stack.ctx.db.session() as s:
        d = Deployment(name="old", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config={}, state="failed")
        s.add(d)
        s.flush()
        s.add(DeploymentEvent(deployment_id=d.id, from_state="starting", to_state="failed", message="x",
                              log_tail=_secret_log()))
        dep_id = d.id
    body = api_stack.get(f"/dgxmm/api/deployments/{dep_id}", headers=h).text
    assert HF_TOKEN not in body and "abcDEF123ghiJKL456mnoPQR" not in body and "No space left" in body


def test_api_masks_a_pem_key_body_in_an_old_stored_tail(api_stack):
    from dgxmm.db import Deployment, DeploymentEvent
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    with api_stack.ctx.db.session() as s:
        d = Deployment(name="old-pem", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config={}, state="failed")
        s.add(d)
        s.flush()
        s.add(DeploymentEvent(deployment_id=d.id, from_state="starting", to_state="failed", message="x",
                              log_tail="\n".join(_PEM_BODY[1:] + ["-----END PRIVATE KEY-----", "ERROR boom"])))
        dep_id = d.id
    body = api_stack.get(f"/dgxmm/api/deployments/{dep_id}", headers=h).text
    assert not any(b in body for b in _PEM_BODY) and "ERROR boom" in body


# A PEM private key printed by an engine (e.g. a misconfigured TLS setup): as the stored tail condenses the log, the
# BEGIN line can end up in the omitted part and the key body in the kept tail, or the whole block in the tail.
_PEM_BODY = ["MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7" + "a" * 12,
             "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe9Dc8Ba7ZYXWVUTSRQPONMLKJIH",
             "q1W2e3R4t5Y6u7I8o9P0a1S2d3F4g5H6j7K8l9Z0x1C2v3B4n5M6q7W8",
             "e3R4t5Y6u7I8o9P0a1S2d3F4g5H6j7K8l9Z0x1C2v3B4n5M6q7W8e9R0",
             "Ab12Cd34=="]


def test_stored_tail_masks_a_pem_key_cut_by_the_tail_and_a_whole_block():
    from dgxmm.deployer.logtail import stored_tail
    cut = (["INFO start"] * 50 + ["-----BEGIN PRIVATE KEY-----"] + _PEM_BODY + ["-----END PRIVATE KEY-----"]
           + [f"INFO step {i}" for i in range(TAIL_LINES - 3)])  # the BEGIN line and two body lines are cut
    whole = ["-----BEGIN RSA PRIVATE KEY-----", *_PEM_BODY, "-----END RSA PRIVATE KEY-----", "ERROR boom"]
    for text in ("\n".join(cut), "\n".join(whole)):
        out = stored_tail(text)
        assert not any(b in out for b in _PEM_BODY), out
    assert "ERROR boom" in stored_tail("\n".join(whole))


async def test_failed_deployment_never_stores_a_pem_key_body(stack):
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text = "\n".join(["-----BEGIN PRIVATE KEY-----", *_PEM_BODY, "-----END PRIVATE KEY-----"]
                           + [f"(EngineCore) frame {i}" for i in range(TAIL_LINES - 20)])
    c.status, c.exit_code = "exited", 1
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    tail = stack.deployer.view(dep["id"], with_events=True)["events"][0]["log_tail"]
    assert not any(b in tail for b in _PEM_BODY) and "frame 179" in tail
