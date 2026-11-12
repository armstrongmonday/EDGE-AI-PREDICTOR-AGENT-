"""Build the desktop bundle with PyInstaller (run on the OS you are targeting: Windows .exe must be built on Windows).
    pip install . pyinstaller && python packaging/build_desktop.py   ->  dist/EdgeAgent/"""
import PyInstaller.__main__ as pi

pi.run(["packaging/desktop_entry.py", "--noconfirm", "--clean", "--onedir", "--name", "EdgeAgent",
        "--collect-data", "edgeagent", "--collect-submodules", "edgeagent",
        "--collect-all", "duckdb", "--collect-submodules", "sklearn", "--collect-data", "sklearn",
        "--hidden-import", "uvicorn.logging", "--hidden-import", "uvicorn.loops.auto", "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets.auto", "--hidden-import", "uvicorn.lifespan.on",
        "--exclude-module", "tkinter", "--exclude-module", "matplotlib"])
