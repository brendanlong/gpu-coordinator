"""Takes web-dashboard.png and web-logs.png of `demo_state`'s fleet.

From the repo root: `uv run --with playwright python docs/media/screenshot_web.py`
(the first time, `uv run --with playwright playwright install chromium`).
Needs `optipng` on PATH.
"""

from __future__ import annotations

import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from demo_state import Handler
from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]

HERE = Path(__file__).parent


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    dashboard, logs = HERE / "web-dashboard.png", HERE / "web-logs.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 660})
        page.emulate_media(color_scheme="dark")
        page.goto(url)
        page.get_by_text("a100-burst").first.wait_for()
        page.screenshot(path=dashboard, full_page=True)

        page.get_by_role("button", name="Logs").first.click()
        page.locator("#log-panel").get_by_text("uploaded checkpoints").wait_for()
        page.check("#log-follow")
        page.locator("#log-panel").click(position={"x": 5, "y": 5})
        page.screenshot(path=logs)
        browser.close()
    server.shutdown()
    subprocess.run(["optipng", "-quiet", "-o3", dashboard, logs], check=True)


if __name__ == "__main__":
    main()
