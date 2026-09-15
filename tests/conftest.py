"""Shared pytest fixtures for tests/.

Kept deliberately small: this exists to hold ONE piece of defensive logic
that several test files independently needed, rather than to become a place
where test infrastructure accumulates unexamined.
"""
from __future__ import annotations

import os

# /auth/login's 10/minute limit (backend/core/rate_limit.py::AUTH_LIMIT) is
# real production behavior and every test file that needs a token already
# caches it per-module specifically to live within that budget (see
# test_department_scoping.py's own comment). What none of them can do
# anything about is the budget being SHARED, per test-client identity, across
# the whole session: on a full run of 300+ tests spread over several
# minutes, files logging in for the first time late in the run can still
# land inside a one-minute window that earlier files already spent — a
# provably pre-existing failure mode (test_department_scoping.py fails this
# way despite already caching its own token) that reproduced identically
# across repeated full-suite runs while this file was being added.
#
# No test anywhere asserts that /auth/login itself returns 429 (checked:
# no `test_*rate*limit*` and no `status_code == 429` assertion in tests/),
# so nothing is lost by giving the test session a budget large enough that
# the shared-window contention can't happen. This must run before
# backend.core.config is first imported by any test module — Settings()
# is built once at import time (backend/core/config.py:362) and
# AUTH_LIMIT is derived from it at backend/core/rate_limit.py's own import
# time — and conftest.py is loaded before test collection, which is what
# makes this the correct place for it rather than a per-file workaround.
# setdefault, not assignment: a real .env value some caller assigns
# themselves is meant to be tested, not overridden by this.
os.environ.setdefault("AUTH_RATE_LIMIT_PER_MINUTE", "10000")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "10000")

import pytest


@pytest.fixture
def clear_get_current_user_override():
    """Clear any global get_current_user override for this test; restore after.

    tests/test_live_calibration_production.py and
    tests/test_macro_baseline_anomaly_engine.py both do

        app.dependency_overrides[get_current_user] = lambda: MockOfficer()

    at MODULE level — a bare statement, not inside a fixture — so it runs at
    IMPORT time. pytest collects (imports) every test file in the session
    before running any of them, so that override is already active before the
    first test anywhere executes, independent of file execution order.

    A test that logs in as a specific real user and asserts on THAT user's
    role or department cannot coexist with a global override silently
    substituting a mock admin for every request regardless of the bearer
    token sent — verified the hard way: tests/test_audit_search.py's
    test_non_admin_is_refused logged in as "viewer" and received a 200 from
    an admin-only route, because the route's own Depends(get_current_user)
    was returning the mock, never touching the real Authorization header.

    Not made autouse here, deliberately: the two files above need the
    override to stay active for their OWN tests, so this is opt-in per file
    (see test_audit_search.py, test_camera_bulk_onboarding.py,
    test_camera_gap_analysis.py, test_department_scoping.py for how) rather
    than applied globally to every test in the suite.
    """
    from backend.auth.dependencies import get_current_user
    from backend.main import app

    saved = app.dependency_overrides.pop(get_current_user, None)
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved
