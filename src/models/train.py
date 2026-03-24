"""Backward-compatible entrypoint; full pipeline lives in ``src.train``."""

from __future__ import annotations

from src.train import main

if __name__ == "__main__":
    main()
