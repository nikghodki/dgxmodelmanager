"""Capture the documentation screenshots (docs/images/) from the released UI against made-up demo data.

The built frontend (copied out of the DGXMM image by run.sh) is served by Playwright route handlers and every API
request is answered from demo_data.py, so no DGXMM backend, engine, GPU or network is involved and every run gives
the same pictures. Run it through run.sh; see docs/README.md."""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, Route, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_data  # noqa: E402
from shots import SHOTS  # noqa: E402

ORIGIN = "http://dgxmm.demo"  # never resolved: the routes below answer every request
BASE = "/dgxmm"

# EventSource is replaced by a silent stub (the demo has no live stream); the clock is fixed for stable "x min ago".
INIT_JS = """
window.EventSource = class { constructor() { this.readyState = 1; } addEventListener() {} removeEventListener() {} close() {} };
localStorage.setItem('dgxmm-theme', %s);
document.addEventListener('DOMContentLoaded', () => {
  const s = document.createElement('style');  // no caret blink or transition half-way in a capture
  s.textContent = '*, *::before, *::after { caret-color: transparent !important; transition: none !important; }';
  document.head.appendChild(s);
});
"""


def static_handler(static: Path):
    def handle(route: Route) -> None:
        path = urlparse(route.request.url).path
        if not path.startswith(BASE + "/"):
            route.abort()
            return
        rel = path[len(BASE) + 1:]
        f = (static / rel).resolve()
        if rel and f.is_file() and static.resolve() in f.parents:
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            route.fulfill(status=200, body=f.read_bytes(), headers={"content-type": ctype})
        else:  # an app route: the SPA's index.html
            route.fulfill(status=200, body=(static / "index.html").read_bytes(), headers={"content-type": "text/html"})
    return handle


def api_handler(world: demo_data.World, missing: set):
    def handle(route: Route) -> None:
        req = route.request
        u = urlparse(req.url)
        path = u.path[len(BASE) + len("/api"):]
        query = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = None
        if req.post_data:
            try:
                body = json.loads(req.post_data)
            except ValueError:
                body = None
        try:
            status, data = world.handle(req.method, path, query, body)
        except Exception:  # a bug in the demo data: report it, answer 500
            traceback.print_exc()
            missing.add(f"{req.method} {path} (handler error)")
            status, data = 500, {"detail": "demo data error"}
        if status == 404 and data is demo_data.MISSING:
            missing.add(f"{req.method} {path}")
            data = {"detail": {"code": "not_found", "message": "not in the demo data"}}
        if status == 204:
            route.fulfill(status=204, body="")
        else:
            route.fulfill(status=status, body=json.dumps(data), headers={"content-type": "application/json"})
    return handle


def take(page: Page, shot: dict, out: Path, theme: str) -> Path:
    name = shot["name"] if shot.get("single") else f"{shot['name']}-{theme}"
    dest = out / f"{name}.png"
    target = shot.get("element")
    if target and shot.get("clip_bottom"):  # the last N px of the element (e.g. the end of a long card)
        box = page.locator(target).first.bounding_box()
        h = min(shot["clip_bottom"], box["height"])
        page.screenshot(path=str(dest), animations="disabled", full_page=True,
                        clip={"x": box["x"], "y": box["y"] + box["height"] - h, "width": box["width"], "height": h})
    elif target:
        page.locator(target).first.screenshot(path=str(dest), animations="disabled")
    else:
        if shot.get("full_page"):
            # grow the viewport to the page (the sidebar is viewport-high and sticky, so a plain full-page capture would
            # cut it off); repeat until the height settles
            for _ in range(3):
                h = page.evaluate("document.documentElement.scrollHeight")
                if h == page.viewport_size["height"]:
                    break
                page.set_viewport_size({"width": page.viewport_size["width"], "height": h})
                page.wait_for_timeout(500)
        page.screenshot(path=str(dest), animations="disabled")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True, type=Path, help="the built frontend (the image's /app/static)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="shot names to capture (default: all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    missing: set[str] = set()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for shot in SHOTS:
            if args.only and shot["name"] not in args.only:
                continue
            for theme in shot.get("themes", ("light", "dark")):
                vp = shot.get("viewport", (1440, 900))
                ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=shot.get("scale", 1),
                                          color_scheme=theme, locale="en-US", timezone_id="UTC",
                                          is_mobile=shot.get("mobile", False), has_touch=shot.get("mobile", False))
                ctx.add_init_script(INIT_JS % json.dumps(theme))
                world = demo_data.World(shot.get("scenario", "default"))
                # the route added last is tried first: the API before the static files
                ctx.route(f"{ORIGIN}/**", static_handler(args.static))
                ctx.route(f"{ORIGIN}{BASE}/api/**", api_handler(world, missing))
                page = ctx.new_page()
                page.clock.set_fixed_time(demo_data.NOW)
                page.goto(f"{ORIGIN}{BASE}{shot['path']}")
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:  # polling pages may never be idle for 500 ms
                    pass
                for step in shot.get("steps", ()):
                    step(page)
                page.wait_for_timeout(shot.get("settle_ms", 1500))
                dest = take(page, shot, args.out, theme)
                print(f"{dest.name}: {dest.stat().st_size} bytes")
                ctx.close()
        browser.close()
    if missing:
        print("not in the demo data:", *sorted(missing), sep="\n  ", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
