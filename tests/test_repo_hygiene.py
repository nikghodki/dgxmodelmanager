"""Repository hygiene: tracked files must not identify the development box or its owner (user name, host name,
container names, Tailscale / LAN addresses) or carry names that must stay out of this repository. The identifiers
are stored only as SHA-256 hashes, so this file does not contain them; private addresses are found by range.
Use neutral placeholders instead: user `dgxuser`, host `dgx-host`, documentation addresses 192.0.2.0/24,
198.51.100.0/24, 203.0.113.0/24."""
from __future__ import annotations

import hashlib
import ipaddress
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# sha256 of the lower-cased identifier -> its length (so only tokens of that length are hashed)
FORBIDDEN = {
    "1a97485e9745eace8d5b5bc5242e75dfea1c47f132f686d696d501eba3b58566": 13,
    "2d0faa9c618208f30d4a20ac277fbbf9d26f02fc42dac5a4c2c10b44a4ee5e89": 10,
    "cce4e65071238e7a8ed5e6b03725e16b5451d817e4bb06f332c4d61504aded3b": 16,
    "34119a5330025362afd51e5038d185941fd67f7e4bc8dd1fd49bb5d9f9967f1e": 7,
    "e73b79a0b10f8cdb6ac7dbe4c0a5e25776e1148784b86cf98f7d6719d472af69": 5,
    "7ab7b4d3c0e9d789ad35e357b0e85e966eb8d1baa13df790852227c04b0a116c": 4,
    "eb895d08402e4f6ea7e6588b243a582108194fbd5902fb06456902871d37f4e2": 6,
    "16bc4ce9ceafddfcf5630d049c05afcb702fe33d96e5bf2a6cce34ad88da28be": 8,
    "9748ec7a20a6dff8a1cb9957aa1e30ad34997bdd9d8add618fe663d3830abab0": 25,
    "66271ba1dbf259a9d9b2a8e73c9dd5fca52ca3905074b7ea55ece03222d261c0": 10,
    "0847e72125637326ac42babab801e6c1fa2cdf7040faf03fe95686c82aa28dc5": 12,
}
LENGTHS = set(FORBIDDEN.values())
MAX_PARTS = 3
# Tailscale CGNAT and LAN ranges: real hosts there identify a network. Docker's 172.16.0.0/12 bridges are generic.
PRIVATE = [ipaddress.ip_network(n) for n in ("100.64.0.0/10", "192.168.0.0/16", "10.0.0.0/8")]
_WORDS = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_PARTS = re.compile(r"[a-z0-9]+|[-_.]")
_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?:/(\d{1,2}))?(?![\d.]*\d)")


def tracked_text_files() -> list[tuple[str, str]]:
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    out = []
    for name in names.decode().split("\0"):
        p = ROOT / name
        if not name or not p.is_file() or p.is_symlink():
            continue
        try:
            out.append((name, p.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue  # binary
    return out


def candidates(text: str):
    """Every word of the text and every run of up to MAX_PARTS of its -_. separated parts."""
    for word in _WORDS.findall(text.lower()):
        parts = _PARTS.findall(word)  # alternating part, sep, part, ...
        words = parts[::2]
        for i in range(len(words)):
            for j in range(i + 1, min(i + MAX_PARTS, len(words)) + 1):
                s = "".join(parts[2 * i:2 * j - 1])
                if len(s) in LENGTHS:
                    yield s


def forbidden_hits(text: str) -> set[str]:
    return {h for c in candidates(text) if (h := hashlib.sha256(c.encode()).hexdigest()) in FORBIDDEN}


def private_hosts(text: str) -> list[str]:
    out = []
    for m in _IPV4.finditer(text):
        try:
            ip = ipaddress.ip_address(m.group(1))
        except ValueError:
            continue
        if not any(ip in n for n in PRIVATE):
            continue
        if m.group(2) is not None:  # CIDR notation: fine when it names a network (10.0.0.0/8), not a host
            try:
                if ipaddress.ip_network(f"{ip}/{m.group(2)}", strict=True):
                    continue
            except ValueError:
                pass
        out.append(m.group(0))
    return out


def _ip(*octets: int) -> str:  # built at run time, so this file holds no address literal it would flag
    return ".".join(map(str, octets))


def test_the_checks_work():
    assert forbidden_hits("dgxuser@dgx-host 192.0.2.82") == set()
    lan, ts, ten = _ip(192, 168, 1, 7), _ip(100, 100, 1, 2), _ip(10, 1, 2, 3)
    assert private_hosts(f"ssh -L 2222:localhost:22 u@{lan} and {ts}, {ten}:8000") == [lan, ts, ten]
    assert private_hosts(f"trusted_proxies: {_ip(10, 0, 0, 0)}/8,{_ip(192, 168, 0, 0)}/16; version 10.4.20; "
                         f"1.{_ip(10, 0, 0, 1)}") == []
    assert private_hosts(f"{_ip(10, 0, 0, 5)}/24") == [f"{_ip(10, 0, 0, 5)}/24"]


def test_no_box_identifiers_or_forbidden_names_in_tracked_files():
    bad = {name: sorted(h[:12] for h in hits) for name, text in tracked_text_files() if (hits := forbidden_hits(text))}
    assert bad == {}, f"identifying or forbidden names (sha256 prefixes) in {bad}"


def test_no_tailscale_or_lan_addresses_in_tracked_files():
    bad = {name: hits for name, text in tracked_text_files() if (hits := private_hosts(text))}
    assert bad == {}, f"use documentation addresses (192.0.2.x, 198.51.100.x, 203.0.113.x) instead: {bad}"
