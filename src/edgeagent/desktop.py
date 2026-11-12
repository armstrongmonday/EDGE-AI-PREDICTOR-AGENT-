"""Desktop launcher (Windows/macOS/Linux): starts the engine + API on localhost and opens the app in your browser.
Settings live in a plain config.env in your user folder. Keys never leave your machine except to the data providers."""
from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

APP = "EdgeAgent"
TEMPLATE = """# EdgeAgent settings (restart the app after editing).
# synthetic = built-in simulator for demos only.  live = real data, needs BOTH keys below.
EDGEAGENT_SOURCE=synthetic
API_FOOTBALL_KEY=
ODDS_API_KEY=
EDGEAGENT_LEAGUES=EPL,LALIGA,SERIEA,BUNDES,LIGUE1
ODDS_REGIONS=eu,uk,us,au
"""

def data_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / APP if os.name == "nt" else Path.home() / ".edgeagent"
    base.mkdir(parents=True, exist_ok=True)
    return base

def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())   # real environment variables win

def free_port(start: int) -> int:
    for p in range(start, start + 30):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port found")

def main() -> None:
    d = data_dir()
    cfg = d / "config.env"
    if not cfg.exists():
        cfg.write_text(TEMPLATE, encoding="utf-8")
    load_env(cfg)
    os.environ.setdefault("EDGEAGENT_DB", str(d / "edgeagent.duckdb"))
    os.environ.setdefault("EDGEAGENT_AUTORUN", "1")
    if os.environ.get("EDGEAGENT_SOURCE", "synthetic") == "synthetic":
        os.environ.setdefault("EDGEAGENT_SIM_STEP_HOURS", "3")
    port = free_port(int(os.environ.get("EDGEAGENT_PORT", "8000")))
    import uvicorn  # imported only now: edgeagent.config reads the environment at import time

    from edgeagent.api.main import create_app
    try:
        app = create_app()
    except RuntimeError as e:
        print(f"\nCannot start: {e}\nEdit {cfg} and restart.\n")
        input("Press Enter to close…")
        sys.exit(1)
    url = f"http://127.0.0.1:{port}/app/"
    print(f"{APP} running at {url}\nSettings file: {cfg}\nData folder:   {d}\nPress Ctrl+C or close this window to stop.")
    if not os.environ.get("EDGEAGENT_NO_BROWSER"):
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
