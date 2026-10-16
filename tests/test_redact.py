"""Command-line redaction for the processes view (spec §5). Secrets must never reach the client."""
import shlex

import pytest

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
    assert "'Authorization: ***'" in out and "http://x/v1/models" in out
    out = cmd(["sh", "-c", "curl -H 'authorization: bearer tok123456789' http://x"])
    assert "authorization: ***" in out and "http://x" in out
    assert cmd(["app", "--auth-header-value", "x", "Bearer tok123456789"]).endswith("'Bearer ***'")


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
    for name in ("api-tokens", "hf-tokens", "tokens", "requirepass", "passphrase", "dbpass", "sig", "x-amz-signature",
                 "api-key", "api_key", "apikey", "hf-token", "HF_TOKEN", "use-auth-token", "authorization", "secret",
                 "client-secret", "password", "PGPASSWORD", "passwd", "pass", "credentials", "LITELLM_MASTER_KEY",
                 "tokenizer-secret", "ssl-keyfile"):
        assert is_sensitive_name(name), name
    for name in ("model", "port", "tokenizer", "tokenizer-mode", "max-tokens", "max-num-batched-tokens",
                 "enable-auto-tool-choice", "served-model-name", "passthrough", "host", "config",
                 "tokenizers", "tokenizer-revision", "num-speculative-tokens", "max-new-tokens", "bypass-cache"):
        assert not is_sensitive_name(name), name


def test_trailing_sensitive_flag_and_negative_number_value():
    assert cmd(["svc", "--api-key"]) == "svc --api-key"
    assert cmd(["svc", "--token", "-123"]) == "svc --token ***"


# Reviewer probe cases (fix round 1): every one of these leaked or was at risk. S must never survive.
S = "Zq9SECRETvalue7"
LEAK_CASES = {
    "api-tokens": ["svc", "--api-tokens", S],
    "hf-tokens": ["svc", "--hf-tokens", S],
    "tokens": ["svc", "--tokens", S],
    "dash-leading-secret": ["vllm", "--api-key", "-" + S],
    "dash-leading-eq": ["vllm", "--api-key=-" + S],
    "mysql -p glued": ["mysql", "-uroot", "-p" + S],
    "mysqldump -p glued": ["/usr/bin/mysqldump", "-p" + S, "db"],
    "curl -u": ["curl", "-u", "user:" + S, "http://x"],
    "curl --user=": ["curl", "--user=user:" + S, "http://x"],
    "passphrase": ["gpg", "--passphrase", S],
    "requirepass": ["redis-server", "--requirepass", S],
    "dbpass": ["app", "--dbpass=" + S],
    "Authorization token (github)": ["curl", "-H", "Authorization: token ghp_" + S],
    "Authorization ApiKey": ["curl", "-H", "Authorization: ApiKey " + S],
    "Proxy-Authorization": ["curl", "-H", "Proxy-Authorization: Digest a=" + S],
    "url userinfo slash": ["x", "postgres://u:ab/" + S + "@db/x"],
    "url userinfo at": ["x", "postgres://u:ab@" + S + "@db/x"],
    "url userinfo hash": ["x", "https://u:ab#" + S + "@h/x"],
    "ghp bare": ["x", "ghp_" + S * 3],
    "github_pat bare": ["x", "github_pat_" + S * 3],
    "AKIA bare": ["x", "AKIAABCDEFGHIJKLMNOP"],
    "json escaped quote": ["x", "--hf-overrides", '{"token": "ab\\"' + S + '"}'],
    "json nested": ["x", "--hf-overrides", '{"auth":{"token":"' + S + '"}}'],
    "python list argv": ["python", "-c", "subprocess.run(['x','--api-key','" + S + "'])"],
    "python list argv dq": ["python", "-c", 'run(["x", "--api-key", "' + S + '"])'],
    "sas sig": ["curl", "https://a.blob/x?sv=1&sig=" + S],
    "query token": ["wget", "https://h/x?a=1&token=" + S],
    "query key": ["wget", "https://h/x?key=" + S + "&b=2"],
    "query access_token": ["wget", "https://h/x?access_token=" + S],
    "base64 after --secret": ["x", "--secret", "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" + S],
    "sh -c dash secret": ["sh", "-c", "vllm --api-key -" + S],
    "env X=": ["env", "MYSQL_PWD=" + S, "PGPASSWORD=" + S],
    "Cookie": ["curl", "-H", "Cookie: session=" + S],
    "Cookie in sh": ["sh", "-c", "curl -H 'Cookie: a=1; session=" + S + "' http://x"],
    "-H X-Api-Key": ["curl", "-H", "X-Api-Key: " + S],
    "--header=Authorization": ["curl", "--header=Authorization: Bearer " + S],
    "token json no space": ["x", '{"hf_token":"' + S + '"}'],
    "--key value with spaces in sh": ["sh", "-c", "app --api-key '" + S + " more'"],
    "sh -c --api-key double": ["sh", "-c", 'app --api-key "' + S + '"'],
    "redis-cli -a": ["redis-cli", "-a", S],
    "redis-cli -a in sh": ["sh", "-c", "redis-cli -h redis -a " + S + " ping"],
    "mysql -p in sh": ["bash", "-c", "mysql -uroot -p" + S + " -e 'select 1'"],
    "sshpass -p": ["sshpass", "-p", S, "ssh", "host"],
    "slack token": ["x", "xoxb-" + S + "-" + S],
    # fix round 2
    "sshpass -p glued": ["sshpass", "-p" + S, "ssh", "host"],
    "curl -u glued": ["curl", "-uuser:" + S, "http://x"],
    "htpasswd -b": ["htpasswd", "-b", "/etc/nginx/htpasswd", "admin", S],
    "htpasswd -bc": ["htpasswd", "-bc", "/f", "admin", S],
    "htpasswd -nb": ["htpasswd", "-nb", "admin", S],
    "openssl -k": ["openssl", "enc", "-aes-256-cbc", "-k", S, "-in", "f"],
    "openssl -pass pass:": ["openssl", "enc", "-pass", "pass:" + S],
    "openssl -passin pass:": ["openssl", "rsa", "-passin", "pass:" + S, "-in", "k.pem"],
    "openssl pass: in sh": ["sh", "-c", "openssl pkcs12 -export -passout pass:" + S + " -out x.p12"],
    "os.environ sq": ["python", "-c", "import os; os.environ['API_KEY']='" + S + "'; main()"],
    "os.environ dq": ["python", "-c", 'import os; os.environ["X_TOKEN"] = "' + S + '"'],
    "environ setdefault": ["python", "-c", "os.environ.setdefault('HF_TOKEN', '" + S + "')"],
    "tool flag in python list": ["python", "-c", "subprocess.run(['redis-cli','-a','" + S + "','ping'])"],
    "mysql in python list": ["python", "-c", 'subprocess.run(["mysql", "-p' + S + '", "db"])'],
    "nested sh -c": ["sh", "-c", "sh -c 'redis-cli -a " + S + " ping'"],
    "nested sh -c x3": ["bash", "-c", "sh -c \"sh -c 'mysql -p" + S + " db'\""],
    "shlex glued quotes": ["sh", "-c", "redis-cli -a 'Zq9SEC'RETvalue7 ping"],
    "shlex backslash": ["sh", "-c", "redis-cli -a Zq9SEC\\RETvalue7 ping"],
    "shlex dq pieces": ["sh", "-c", 'app --api-key "Zq9"SECRET"value7" --port 1'],
    "--password:": ["app", "--password:" + S],
    "--password: in sh": ["sh", "-c", "app --password:" + S + " run"],
    "rewritten argv0 title": ["redis-cli -h r -a " + S + " ping"],
    "env prefix in sh": ["sh", "-c", "env A=1 redis-cli -a " + S],
    "sudo mysql in sh": ["sh", "-c", "sudo /usr/bin/mysql -p" + S],
}


@pytest.mark.parametrize("name", list(LEAK_CASES))
def test_reviewer_leak_cases(name):
    out = format_command(LEAK_CASES[name])[0]
    assert S not in out, out
    assert "value7" not in out and "Zq9SEC" not in out, out  # no fragment of a split/quoted secret survives
    for prefix in ("ghp_", "github_pat_", "AKIA", "xoxb-"):
        assert prefix not in out, out


def test_reviewer_cases_keep_the_non_secret_parts():
    assert format_command(["mysql", "-uroot", "-p" + S])[0] == "mysql -uroot -p***"
    assert format_command(["mysql", "-p", "db"])[0] == "mysql -p db"  # bare -p prompts; next arg is the database
    assert format_command(["curl", "-u", "user:" + S, "http://x"])[0] == "curl -u *** http://x"
    assert format_command(["redis-cli", "-a", S, "-p", "6379"])[0] == "redis-cli -a *** -p 6379"
    assert format_command(["vllm", "--api-key", "-" + S, "--port", "1"])[0] == "vllm --api-key *** --port 1"
    assert format_command(["curl", "https://a.blob/x?sv=1&sig=" + S])[0] == "curl 'https://a.blob/x?sv=1&sig=***'"
    out = format_command(["sh", "-c", "redis-cli -h redis -a " + S + " ping"])[0]
    assert out == "sh -c 'redis-cli -h redis -a *** ping'"


def test_curl_dash_k_is_insecure_not_a_key():
    # curl -k is --insecure (boolean): the next arg is the URL and must stay visible
    assert format_command(["curl", "-k", "https://h/x"])[0] == "curl -k https://h/x"


def test_round2_keeps_the_non_secret_parts():
    f = lambda argv: format_command(argv)[0]  # noqa: E731
    assert f(["sshpass", "-p" + S, "ssh", "host"]) == "sshpass -p*** ssh host"
    assert f(["curl", "-uuser:" + S, "http://x"]) == "curl -u*** http://x"
    assert f(["htpasswd", "-b", "/etc/h", "admin", S]) == "htpasswd -b /etc/h admin ***"
    assert f(["htpasswd", "-c", "/etc/h", "admin"]) == "htpasswd -c /etc/h admin"  # prompts: nothing secret
    assert f(["openssl", "rsa", "-passin", "pass:" + S, "-in", "k.pem"]) == "openssl rsa -passin *** -in k.pem"
    assert f(["app", "--password:" + S]) == "app --password:***"
    assert f(["sh", "-c", "redis-cli -a 'Zq9SEC'RETvalue7 ping"]) == "sh -c 'redis-cli -a *** ping'"
    assert f(["sh", "-c", "cd /x && redis-cli -h r -a " + S + " ping; echo ok"]) == \
        "sh -c 'cd /x && redis-cli -h r -a *** ping; echo ok'"
    out = f(["python", "-c", "subprocess.run(['redis-cli','-a','" + S + "','ping'])"])
    assert "'\"'\"'redis-cli'\"'\"'" in out and "ping" in out
    out = f(["sh", "-c", "sh -c 'redis-cli -a " + S + " ping'"])
    assert "redis-cli -a *** ping" in out


# Fix round 3: tool detection after shell keywords and prefix commands (with their options/operands)
ROUND3_CASES = {
    "timeout + abs path": ["sh", "-c", "timeout 10 /usr/bin/sshpass -p" + S + " ssh h"],
    "timeout opts": ["sh", "-c", "timeout -s KILL -k 5 30s /usr/bin/redis-cli -a " + S],
    "timeout argv": ["timeout", "10", "/usr/bin/sshpass", "-p" + S, "ssh", "h"],
    "sudo -u": ["sh", "-c", "sudo -u bob /usr/bin/mysql -p" + S + " db"],
    "sudo -E -u --": ["sudo", "-E", "-u", "bob", "--", "/usr/bin/mysql", "-p" + S],
    "while do": ["sh", "-c", "while true; do /opt/redis/bin/redis-cli -a " + S + " ping; sleep 1; done"],
    "if then": ["sh", "-c", "if [ -x /x ]; then /usr/bin/redis-cli -a " + S + "; fi"],
    "brace group": ["sh", "-c", "{ /usr/local/bin/redis-cli -a " + S + "; }"],
    "subshell": ["sh", "-c", "(/usr/local/bin/redis-cli -a " + S + ")"],
    "and-or": ["sh", "-c", "true && /usr/bin/mysql -p" + S + " || false"],
    "pipe": ["sh", "-c", "echo x | /usr/bin/redis-cli -a " + S],
    "bang": ["sh", "-c", "! /usr/bin/redis-cli -a " + S],
    "env -i VAR": ["sh", "-c", "env -i A=1 B=2 /usr/bin/redis-cli -a " + S],
    "env -u NAME": ["env", "-u", "HOME", "/usr/bin/redis-cli", "-a", S],
    "nice -n": ["sh", "-c", "nice -n 10 /usr/bin/mysql -p" + S],
    "nohup exec": ["sh", "-c", "nohup exec /usr/bin/redis-cli -a " + S],
    "stdbuf ionice chrt taskset setsid": ["sh", "-c", "stdbuf -oL ionice -c 2 -n 7 chrt -f 10 taskset -c 0-3 setsid /usr/bin/redis-cli -a " + S],
    "xargs": ["sh", "-c", "echo k | xargs -n 1 /usr/bin/redis-cli -a " + S],
    "time -p": ["sh", "-c", "time -p /usr/bin/mysql -p" + S],
    "VAR= prefix": ["sh", "-c", "LANG=C /usr/bin/mysql -p" + S],
    "mid-argv abs tool": ["docker", "exec", "c", "/usr/bin/redis-cli", "-a", S],
    # tool families
    "docker login -p": ["docker", "login", "-u", "bob", "-p", S, "registry.example.com"],
    "podman login --password": ["podman", "login", "--password", S, "quay.io"],
    "docker login glued": ["docker", "login", "-p" + S, "r"],
    "mongosh -p": ["mongosh", "mongodb://h", "-u", "admin", "-p", S],
    "mongo -p": ["/usr/bin/mongo", "-p", S],
    "openssl -psk": ["openssl", "s_client", "-psk", "deadbeef" + S, "-connect", "h:443"],
    "lftp -u": ["lftp", "-u", "user," + S, "ftp://h"],
    "smbclient -U": ["smbclient", "//h/share", "-U", "user%" + S],
    "smbclient -U glued": ["smbclient", "//h/share", "-Uuser%" + S],
    "smbclient --user=": ["smbclient", "//h/share", "--user=user%" + S],
    # Python / Perl argv shapes
    "python tuple": ["python", "-c", "subprocess.run(('redis-cli','-a','" + S + "'))"],
    "mixed list": ["python", "-c", "subprocess.run(['redis-cli','-a','" + S + "',port])"],
    "mixed list kw": ["python", "-c", "run(['redis-cli', '-h', host, '-a', '" + S + "'], check=True)"],
    "environb bytes": ["python", "-c", "os.environb[b'API_KEY']=b'" + S + "'"],
    "perl system": ["perl", "-e", "system('redis-cli','-a','" + S + "')"],
    "perl system dq": ["perl", "-e", 'system("mysql", "-p' + S + '", "db")'],
}


@pytest.mark.parametrize("name", list(ROUND3_CASES))
def test_round3_leak_cases(name):
    out = format_command(ROUND3_CASES[name])[0]
    assert S not in out and "value7" not in out and "Zq9SEC" not in out, out


def test_round3_keeps_structure():
    f = lambda argv: format_command(argv)[0]  # noqa: E731
    assert f(["docker", "run", "-p", "8080:80", "nginx"]) == "docker run -p 8080:80 nginx"  # -p is publish here
    assert f(["sudo", "-u", "bob", "ls", "-la"]) == "sudo -u bob ls -la"
    assert f(["timeout", "10", "/usr/bin/sshpass", "-p" + S, "ssh", "h"]) == "timeout 10 /usr/bin/sshpass -p*** ssh h"
    assert f(["sh", "-c", "while true; do /opt/redis/bin/redis-cli -a " + S + " ping; sleep 1; done"]) == \
        "sh -c 'while true; do /opt/redis/bin/redis-cli -a *** ping; sleep 1; done'"
    assert f(["python", "-c", "subprocess.run(['redis-cli','-a','" + S + "',port])"]) == \
        "python -c 'subprocess.run(['\"'\"'redis-cli'\"'\"','\"'\"'-a'\"'\"','\"'\"'***'\"'\"',port])'"


def test_depth_beyond_max_masks_the_remaining_script():
    from dgxmm.memory.redact import MAX_SCRIPT_DEPTH
    inner = "redis-cli -a " + S
    for _ in range(MAX_SCRIPT_DEPTH + 1):
        inner = "sh -c " + shlex.quote(inner)
    out = format_command(["sh", "-c", inner])[0]
    assert S not in out and "***" in out


# Fix round 3: bounded cost. Any local user controls argv; sample() redacts the top 8 every budget tick.
PATHOLOGICAL = {
    "dot-a plain": ["x", "." + ".a" * 32768],
    "dot-a spaced": ["x", ("." + ".a" * 500 + " ") * 64],
    "dot-a sh -c": ["sh", "-c", "." + ".a" * 32768],
    "dot-a sh -c spaced": ["sh", "-c", ("." + ".a" * 500 + " ") * 64],
    "scheme repeat": ["x", "a://" * 16384],
    "scheme repeat spaced": ["sh", "-c", ("a://" * 250 + " ") * 64],
    "x-a-a header": ["x", "x-" + "a-" * 32767],
    "x-a-a spaced": ["sh", "-c", ("x-" + "a-" * 500 + " ") * 64],
    "dash flags": ["sh", "-c", ".-a" * 21845],
    "quotes": ["sh", "-c", 'token="' * 9362],
    "brackets": ["python", "-c", "['a'," * 13107],
    "parens": ["perl", "-e", "('a'," * 13107],
    "many args": ["x"] + ["." + ".a" * 100] * 320,
    "many sh -c levels": ["sh", "-c", "sh -c " * 10000],
}


@pytest.mark.parametrize("name", list(PATHOLOGICAL))
def test_pathological_inputs_redact_quickly(name):
    import time
    argv = PATHOLOGICAL[name]
    assert sum(len(a) for a in argv) >= 60_000
    t0 = time.perf_counter()
    out, _ = format_command(argv)
    assert time.perf_counter() - t0 < 0.2, name  # was 54 s (plain) / 166 s (sh -c) before the bounds
    assert len(out.encode()) <= MAX_COMMAND_BYTES + 3


@pytest.mark.parametrize("text", ["." + ".a" * 32768, "a://" * 16384, "x-" + "a-" * 32767, ".-a" * 21845])
def test_text_rules_are_linear(text):
    import time
    from dgxmm.memory.redact import redact_text
    t0 = time.perf_counter()
    redact_text(text)
    assert time.perf_counter() - t0 < 0.2


def test_input_caps_fail_closed():
    from dgxmm.memory.redact import MAX_TOKEN_CHARS
    # a secret straddling the per-token cut is never partially shown
    tok = "a " * ((MAX_TOKEN_CHARS - 10) // 2) + "--api-key=" + S + "x" * 50
    out = format_command(["sh", "-c", tok])[0]
    assert "Zq9" not in out and "[truncated]" in out
    long_tail = ["x"] + ["arg"] * 5000 + ["--api-key", S]
    out, truncated = format_command(long_tail)
    assert truncated and S not in out


SWEEP_UNITS = [".a", "a://", "x-a", ".-a", "a=", "-", "[", "('a',", 'token="', "'", '"', "\\", "a:", "a@", "Bearer ",
               "pass:", "x-key:", "{", "$(", "a ", "-a ", "--api-key ", "sh -c '", "hf_", "sk-", "eyJ", "%", "=", ":",
               "a]=", "['a']=", "os.environ['", "cookie:", "authorization: ", "a,", '"a",', "b'a',", "do ", "; ",
               "sudo -u ", "env A=1 ", "timeout 1 "]


@pytest.mark.parametrize("unit", SWEEP_UNITS)
def test_repeated_unit_sweep_stays_fast(unit):
    import time
    from dgxmm.memory.redact import redact_text
    t = unit * (65536 // len(unit))
    for fn in (lambda: redact_text(t), lambda: format_command(["x", t]), lambda: format_command(["sh", "-c", t]),
               lambda: format_command(["sh", "-c", " ".join([t[:400]] * 160)]),
               lambda: format_command(["x"] + [t[:200]] * 320)):
        t0 = time.perf_counter()
        fn()
        assert time.perf_counter() - t0 < 0.2, unit  # measured worst ~30 ms on the DGX
