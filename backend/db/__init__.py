"""backend.db — re-exports the legacy sqlite helpers from backend/db.py.

WHY THIS FILE HAS CONTENT
  The repo contains BOTH:
      backend/db.py          the original sqlite3 helper module
                             (init_db, insert_alert, upsert_track, ...)
      backend/db/            this package (models, session, migrations)

  Python resolves a package before a same-named module, so `from backend
  import db` bound to this package - which was empty - and every legacy call
  raised at runtime, e.g.

      AttributeError: module 'backend.db' has no attribute 'insert_alert'

  That is not a startup error. It fires only when the code path is reached,
  so it killed the ANPR aggregator mid-run and took the whole detection
  pipeline thread with it (pipeline_bridge logs "Unhandled exception in
  pipeline thread", the watchdog reports "Dashboard is frozen"). It needs a
  low-confidence plate read to trigger, which is why it surfaced on
  vehicle-heavy daylight clips rather than on quiet night footage.

  Rather than rewrite every legacy call site, load backend/db.py directly by
  path and re-export its public names here. Importing it by module name is
  impossible - that name now refers to this package.

  Longer term the right fix is to merge db.py into this package and delete
  it, so there is one obvious place the database lives.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_legacy_path = _Path(__file__).resolve().parent.parent / "db.py"

if _legacy_path.is_file():
    _spec = _ilu.spec_from_file_location("backend._db_legacy", _legacy_path)
    if _spec and _spec.loader:
        _legacy = _ilu.module_from_spec(_spec)
        # Register before exec so any self-referential import inside db.py
        # resolves to this same object instead of re-entering the package.
        _sys.modules["backend._db_legacy"] = _legacy
        _spec.loader.exec_module(_legacy)

        for _name in dir(_legacy):
            if not _name.startswith("_") and _name not in globals():
                globals()[_name] = getattr(_legacy, _name)

        __all__ = [n for n in dir(_legacy) if not n.startswith("_")]
    else:  # pragma: no cover - only if the file becomes unreadable
        __all__ = []
else:  # pragma: no cover - db.py removed after a future consolidation
    __all__ = []
