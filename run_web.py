#!/usr/bin/env python3
"""
Run the SampleScout web interface.

Usage:
    python run_web.py
    python run_web.py --port 8080
    python run_web.py --host 0.0.0.0 --port 8000
"""

import argparse
import webbrowser
from threading import Timer


def open_browser(port: int):
    """Open browser after server starts."""
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(description="Run SampleScout web interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎵 SampleScout Web Interface                            ║
║                                                           ║
║   Running at: http://{args.host}:{args.port:<5}                        ║
║                                                           ║
║   Press Ctrl+C to stop                                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Open browser after short delay
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        Timer(1.5, lambda: open_browser(args.port)).start()

    # Run server
    import uvicorn
    from src.web.api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
