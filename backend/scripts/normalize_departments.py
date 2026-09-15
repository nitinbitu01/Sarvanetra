"""
backend/scripts/normalize_departments.py — Normalize camera and user departments in DB.

Fixes:
1. Case mismatches: 'Police' -> 'police', 'Transport' -> 'transport'.
2. Trims leading/trailing whitespace.
3. Assigns realistic departments to unassigned (NULL) cameras so RBAC does not leak or drop them.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

DB_PATHS = [Path("output/sentinel.db"), Path("sentinel.db")]


def normalize_departments() -> None:
    for db_path in DB_PATHS:
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "cameras" not in tables:
            conn.close()
            continue

        print(f"==================================================================")
        print(f" [SENTINEL GUJARAT] RBAC & DEPARTMENT NORMALIZATION MIGRATION")
        print(f"==================================================================")
        print(f"Target Database: {db_path.resolve()}\n")

        # 1. Inspect before state
        cur.execute("SELECT id, camera_id, name, department FROM cameras ORDER BY id")
        cams_before = cur.fetchall()
        print(f"[1/4] AUDITING {len(cams_before)} CAMERAS BEFORE MIGRATION:")
        dept_counts_before = Counter(row[3] for row in cams_before)
        for dept, count in dept_counts_before.items():
            print(f"    • department = {repr(dept):<20} : {count} camera(s)")

        # 2. Update NULL and Case-Mismatched Camera departments
        defaults = {
            "CAM_03": "police",         # Ahmedabad - O.N.G.C. Office
            "CAM_04": "traffic_police", # Ahmedabad - Paldi Circle
            "CAM-RT-A": "police",       # Main Entrance Cam
            "CAM-RT-B": "police",       # Parking Lot Cam
        }

        updated_cams = 0
        for cid, cam_code, name, dept in cams_before:
            new_dept = dept
            if not dept or str(dept).strip() == "" or dept == "None":
                new_dept = defaults.get(cam_code, "police")
            else:
                new_dept = str(dept).strip().lower()

            if new_dept != dept:
                cur.execute("UPDATE cameras SET department = ? WHERE id = ?", (new_dept, cid))
                updated_cams += 1

        conn.commit()
        print(f"\n[2/4] NORMALIZATION APPLIED: {updated_cams} camera department(s) normalized/assigned.")

        # 3. Normalize users table departments
        cur.execute("PRAGMA table_info(users)")
        user_cols = [c[1] for c in cur.fetchall()]
        updated_users = 0
        if "department" in user_cols:
            cur.execute("SELECT id, username, role, department FROM users")
            users_before = cur.fetchall()
            for uid, uname, role, udept in users_before:
                if udept and udept.strip() != "":
                    norm_udept = udept.strip().lower()
                    if norm_udept != udept:
                        cur.execute("UPDATE users SET department = ? WHERE id = ?", (norm_udept, uid))
                        updated_users += 1
            conn.commit()
        print(f"[3/4] USER DEPARTMENTS NORMALIZED: {updated_users} user(s) updated.")

        # 4. Verify after state
        cur.execute("SELECT id, camera_id, name, department FROM cameras ORDER BY id")
        cams_after = cur.fetchall()
        dept_counts_after = Counter(row[3] for row in cams_after)

        print(f"\n[4/4] POST-MIGRATION VERIFICATION (AFTER STATE):")
        for dept, count in sorted(dept_counts_after.items(), key=lambda x: -x[1]):
            print(f"    • department = {repr(dept):<20} : {count} camera(s)")

        has_nulls = any(c[3] is None for c in cams_after)
        has_uppercase = any(c[3] and any(char.isupper() for char in c[3]) for c in cams_after)

        print(f"\n==================================================================")
        print(f" INTEGRITY CHECKS:")
        print(f"    • Unassigned/NULL Cameras: {0 if not has_nulls else 'FAILED'} (Target: 0)")
        print(f"    • Mixed-case / Dirty strings: {0 if not has_uppercase else 'FAILED'} (Target: 0)")
        print(f"    • Police Cameras Available: {dept_counts_after.get('police', 0)} (Consolidated & fully accessible)")
        print(f"==================================================================\n")

        conn.close()


if __name__ == "__main__":
    normalize_departments()
