"""Cleanly restart FastAPI backend server on port 8000 using process search."""
import os
import sys
import time
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import psutil

def kill_old_backend():
    print("Searching for old backend server processes...")
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if p.info['pid'] == 3492:
                print(f"Killing PID 3492 ({p.name()})...")
                p.kill()
                time.sleep(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def start_backend():
    print("Starting updated FastAPI backend server on port 8000...")
    py = sys.executable
    log_out = open(ROOT / "output" / "backend.log", "w", encoding="utf-8")
    log_err = open(ROOT / "output" / "backend.err", "w", encoding="utf-8")
    
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--host", "127.0.0.1"],
        cwd=str(ROOT),
        stdout=log_out,
        stderr=log_err,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    print(f"Backend launched with PID {proc.pid}.")

if __name__ == "__main__":
    kill_old_backend()
    start_backend()
