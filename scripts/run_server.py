import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    loop_str = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        loop=loop_str,
        http="h11",
        timeout_keep_alive=30,
        app_dir=str(ROOT),
    )
