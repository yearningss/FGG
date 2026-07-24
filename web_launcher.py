"""
FGG Web Dashboard Launcher.
Launches the web application in browser.
"""

from __future__ import annotations

from fgg.web.app import start_web_server

if __name__ == "__main__":
    start_web_server(port=8080, open_browser=True)
