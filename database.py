"""
database.py
SQLite storage for violation records. Handles inserts, queries for the
evidence log/analytics tabs, and repeat-offender detection by plate number.
"""

import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = "violations.db"


def get_connection():
    """Returns a new SQLite connection. Call this per-operation to keep
    things simple and thread-safe for Streamlit's execution model."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the violations table if it doesn't already exist.
    Call this once at app startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity INTEGER NOT NULL,
            zone TEXT,
            plate_number TEXT,
            plate_confidence REAL,
            reason TEXT,
            evidence_image_path TEXT,
            challan_id TEXT,
            challan_pdf_path TEXT,
            source TEXT DEFAULT 'camera'
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {Path(DB_PATH).resolve()}")


def insert_violation(violation_type, confidence, severity, zone=None,
                      plate_number=None, plate_confidence=None, reason=None,
                      evidence_image_path=None, challan_id=None,
                      challan_pdf_path=None, source="camera"):
    """
    Inserts a single violation record. Returns the new row's id.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO violations (
            timestamp, violation_type, confidence, severity, zone,
            plate_number, plate_confidence, reason, evidence_image_path,
            challan_id, challan_pdf_path, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        violation_type, confidence, severity, zone,
        plate_number, plate_confidence, reason, evidence_image_path,
        challan_id, challan_pdf_path, source
    ))

    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_violations(limit=500):
    """Returns all violation records, most recent first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violations_by_type(violation_type):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM violations WHERE violation_type = ? ORDER BY timestamp DESC",
        (violation_type,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violation_counts_by_type():
    """Returns {violation_type: count} for the analytics pie chart."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT violation_type, COUNT(*) as count
        FROM violations
        GROUP BY violation_type
        ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row["violation_type"]: row["count"] for row in rows}


def get_violation_counts_by_zone():
    """Returns {zone: total_severity} for the zone heatmap/risk ranking."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT zone, COUNT(*) as count, SUM(severity) as total_severity
        FROM violations
        WHERE zone IS NOT NULL
        GROUP BY zone
        ORDER BY total_severity DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def check_repeat_offender(plate_number, min_violations=3):
    """
    Checks if a plate number has 3+ prior violations logged.
    Returns (is_repeat_offender: bool, violation_count: int, past_records: list)
    """
    if not plate_number:
        return False, 0, []

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM violations WHERE plate_number = ? ORDER BY timestamp DESC",
        (plate_number,)
    )
    rows = cursor.fetchall()
    conn.close()

    records = [dict(row) for row in rows]
    count = len(records)
    is_repeat = count >= min_violations

    return is_repeat, count, records


def get_repeat_offenders(min_violations=3):
    """
    Returns a list of {plate_number, violation_count} for all plates with
    min_violations or more recorded violations. Used in the shift report.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT plate_number, COUNT(*) as violation_count
        FROM violations
        WHERE plate_number IS NOT NULL AND plate_number != ''
        GROUP BY plate_number
        HAVING COUNT(*) >= ?
        ORDER BY violation_count DESC
    """, (min_violations,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violations_by_hour():
    """
    Returns {hour_of_day (0-23): count} for the hourly heatmap chart.
    Parses the ISO timestamp stored in each record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM violations")
    rows = cursor.fetchall()
    conn.close()

    hour_counts = {h: 0 for h in range(24)}
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["timestamp"])
            hour_counts[dt.hour] += 1
        except (ValueError, TypeError):
            continue

    return hour_counts


def get_summary_stats():
    """Returns overall counts used for the dashboard header / shift report."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM violations")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(DISTINCT plate_number) as unique_plates FROM violations WHERE plate_number IS NOT NULL")
    unique_plates = cursor.fetchone()["unique_plates"]

    cursor.execute("SELECT AVG(severity) as avg_severity FROM violations")
    avg_severity = cursor.fetchone()["avg_severity"]

    conn.close()

    return {
        "total_violations": total,
        "unique_plates_flagged": unique_plates,
        "avg_severity": round(avg_severity, 2) if avg_severity else 0,
    }


if __name__ == "__main__":
    # Standalone test — creates the DB and inserts a few sample records
    init_db()

    print("\nInserting test records...")
    insert_violation(
        violation_type="helmet_violation",
        confidence=0.91,
        severity=8,
        zone="MG Road",
        plate_number="KA01AB1234",
        plate_confidence=0.85,
        reason="Test record - rider without helmet",
        source="camera",
    )
    insert_violation(
        violation_type="helmet_violation",
        confidence=0.88,
        severity=8,
        zone="Silk Board",
        plate_number="KA01AB1234",  # same plate, second violation
        plate_confidence=0.80,
        reason="Test record - rider without helmet, second instance",
        source="camera",
    )
    insert_violation(
        violation_type="illegal_parking",
        confidence=0.75,
        severity=4,
        zone="MG Road",
        plate_number="KA05XY9999",
        plate_confidence=0.70,
        reason="Test record - parked in no-parking zone",
        source="citizen_report",
    )

    print("\nAll violations:")
    for v in get_all_violations():
        print(f"  [{v['id']}] {v['violation_type']} | plate={v['plate_number']} | zone={v['zone']}")

    print("\nCounts by type:")
    print(get_violation_counts_by_type())

    print("\nCounts by zone:")
    print(get_violation_counts_by_zone())

    print("\nRepeat offender check for KA01AB1234:")
    is_repeat, count, records = check_repeat_offender("KA01AB1234", min_violations=2)
    print(f"  is_repeat_offender={is_repeat}, count={count}")

    print("\nSummary stats:")
    print(get_summary_stats())

    print("\nDatabase test complete. violations.db created in project root.")
    print("You can delete violations.db and re-run this script anytime to reset test data.")