"""The documentation screenshots: one entry per image, captured in light and dark unless "themes" says otherwise.

name: file name (name-light.png / name-dark.png, or name.png with single=True); path: the app route;
viewport: (width, height) CSS px; full_page: the whole page; element: a selector to capture instead of the page;
scenario: demo_data.World scenario; steps: page actions before the capture."""
from __future__ import annotations


def _select(page, trigger_label: str, option: str) -> None:
    page.get_by_role("combobox", name=trigger_label).click()
    page.get_by_role("option", name=option, exact=True).click()


def wizard_speculative(page) -> None:
    page.locator("[data-speculative] [role=combobox]").first.click()
    page.get_by_role("option", name="draft_model", exact=True).click()
    page.wait_for_selector("[data-draft-suggestion]")


def scroll_to_placement(page) -> None:
    page.get_by_role("radiogroup", name="Where to run").wait_for()
    page.mouse.wheel(0, 5000)


def pick_dgx1(page) -> None:
    page.get_by_role("radiogroup", name="Where to run").get_by_role("radio").first.click()
    page.get_by_text("Ways to make it fit").first.wait_for()


def remote_code(page) -> None:
    page.get_by_role("button", name="Show advanced settings").click()
    page.get_by_role("switch", name="Trust remote code").click()
    page.get_by_text("Code scan:").first.wait_for()


def catalog_details(page) -> None:
    box = page.get_by_role("textbox", name="Search Hugging Face")
    box.fill("Qwen3-8B")
    box.press("Enter")
    page.get_by_role("button", name="Qwen/Qwen3-8B", exact=True).click()
    page.get_by_text("Other variants").wait_for()


def command_ask(page) -> None:
    page.get_by_role("button", name="Search or run a command").click()
    page.keyboard.type("put the qwen chat model to sleep")
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")
    page.get_by_text("Interpretations").wait_for()


SHOTS = [
    {"name": "dashboard", "path": "/", "viewport": (1440, 960)},
    {"name": "dashboard-mobile", "path": "/", "viewport": (390, 844), "scale": 2, "mobile": True, "single": True,
     "themes": ("light",)},
    {"name": "deployments", "path": "/deployments", "viewport": (1440, 900)},
    {"name": "deployment-detail", "path": "/deployments/3", "full_page": True},
    {"name": "deployment-schedules", "path": "/deployments/4", "full_page": True},
    {"name": "deployment-cluster", "path": "/deployments/7", "full_page": True},
    {"name": "deploy-wizard", "path": "/deploy?model=Qwen%2FQwen3-8B", "scenario": "before-cluster", "full_page": True,
     "steps": [wizard_speculative]},
    {"name": "deploy-placement", "path": "/deploy?model=deepseek-ai%2FDeepSeek-V4-Flash", "scenario": "before-cluster",
     "viewport": (1440, 900), "steps": [scroll_to_placement]},
    {"name": "plan-chooser", "path": "/deploy?model=Qwen%2FQwen3-30B-A3B-Instruct-2507", "viewport": (1440, 1400),
     "steps": [pick_dgx1], "element": "div.lg\\:sticky"},
    {"name": "deploy-remote-code", "path": "/deploy?model=example-org%2Fphi3-custom-code-demo", "viewport": (1440, 1800),
     "steps": [remote_code], "element": "div.shadow-card:has-text('Sizing and features')", "clip_bottom": 623},
    {"name": "insights", "path": "/insights", "full_page": True},
    {"name": "litellm", "path": "/litellm", "full_page": True},
    {"name": "decision-model", "path": "/decision-model", "viewport": (1440, 2420)},
    {"name": "models", "path": "/models", "viewport": (1440, 900)},
    {"name": "processes", "path": "/processes", "viewport": (1440, 900)},
    {"name": "catalog", "path": "/catalog", "viewport": (1440, 960), "steps": [catalog_details]},
    {"name": "command-ask", "path": "/", "viewport": (1440, 900), "steps": [command_ask]},
]
