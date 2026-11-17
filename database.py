import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

# Pakai folder yang sama dengan file ini, sehingga cctv_log.db lama (tabel `faces`) tetap terbaca.
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctv_log.db")

NO_PLATE = "???"
VEHICLE_TYPES = ("Mobil", "Motor")
DIRECTIONS = ("masuk", "keluar")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_setting(key, default=None):
    with _db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(key, value):
    with _db() as conn:
        conn.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value)))


@contextmanager
def _db():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        # Tabel wajah terdaftar (skema sama dengan versi lama)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS faces
               (id INTEGER PRIMARY KEY, name TEXT, encoding TEXT)"""
        )
        # Log gabungan kendaraan + orang
        conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts TEXT NOT NULL,
                   category TEXT NOT NULL,
                   label TEXT NOT NULL,
                   direction TEXT,
                   plate TEXT,
                   line_name TEXT,
                   count INTEGER NOT NULL DEFAULT 1,
                   source TEXT NOT NULL DEFAULT 'auto',
                   note TEXT)"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_cat ON events(category, direction)"
        )
        # Garis virtual buatan pengguna (koordinat ternormalisasi 0..1)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS lines (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL,
                   x1 REAL NOT NULL, y1 REAL NOT NULL,
                   x2 REAL NOT NULL, y2 REAL NOT NULL,
                   flip INTEGER NOT NULL DEFAULT 0,
                   created TEXT NOT NULL)"""
        )


# ----------------------------------------------------------------- events
def add_event(
    category,
    label,
    direction=None,
    plate=None,
    line_name=None,
    count=1,
    source="auto",
    note=None,
    ts=None,
):
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO events
               (ts, category, label, direction, plate, line_name, count, source, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts or _now(), category, label, direction, plate, line_name, int(count), source, note),
        )
        return cur.lastrowid


def update_event_plate(event_id, plate):
    with _db() as conn:
        cur = conn.execute(
            "UPDATE events SET plate = ? WHERE id = ? AND category = 'vehicle'",
            (plate, event_id),
        )
        return cur.rowcount > 0


def delete_event(event_id):
    with _db() as conn:
        cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return cur.rowcount > 0


def _escape_like(text):
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_where(
    category="all",
    vehicle_type="all",
    direction="all",
    search="",
    date_from=None,
    date_to=None,
):
    where, params = [], []
    if category in ("vehicle", "person"):
        where.append("category = ?")
        params.append(category)
    if vehicle_type in VEHICLE_TYPES:
        where.append("(category = 'vehicle' AND label = ?)")
        params.append(vehicle_type)
    if direction in DIRECTIONS:
        where.append("direction = ?")
        params.append(direction)
    if search:
        like = f"%{_escape_like(search)}%"
        where.append(
            "(label LIKE ? ESCAPE '\\' OR plate LIKE ? ESCAPE '\\' "
            "OR line_name LIKE ? ESCAPE '\\' OR note LIKE ? ESCAPE '\\')"
        )
        params += [like] * 4
    if date_from:
        where.append("ts >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where.append("ts <= ?")
        params.append(f"{date_to} 23:59:59")
    sql = ("WHERE " + " AND ".join(where)) if where else ""
    return sql, params


def query_events(sort="newest", limit=100, offset=0, **filters):
    """Kembalikan (baris, total) sesuai filter. sort: 'newest' | 'oldest'."""
    where_sql, params = _build_where(**filters)
    order = "ASC" if sort == "oldest" else "DESC"
    with _db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM events {where_sql}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM events {where_sql} ORDER BY ts {order}, id {order} LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)],
        ).fetchall()
    return [dict(r) for r in rows], total


def get_stats(scope="today"):
    since = datetime.now().strftime("%Y-%m-%d 00:00:00") if scope == "today" else None

    def scoped(base):
        if since:
            return base + " AND ts >= ?", [since]
        return base, []

    stats = {
        "scope": scope,
        "masuk": {t: 0 for t in VEHICLE_TYPES},
        "keluar": {t: 0 for t in VEHICLE_TYPES},
    }
    with _db() as conn:
        sql, params = scoped("SELECT label, direction, SUM(count) FROM events WHERE category = 'vehicle'")
        for label, direction, total in conn.execute(sql + " GROUP BY label, direction", params):
            if label in VEHICLE_TYPES and direction in DIRECTIONS:
                stats[direction][label] += int(total or 0)

        sql, params = scoped(
            "SELECT COALESCE(SUM(count), 0) FROM events "
            "WHERE category = 'vehicle' AND source = 'manual' AND direction = 'masuk'"
        )
        manual_masuk = int(conn.execute(sql, params).fetchone()[0])

        sql, params = scoped("SELECT label, COUNT(*) FROM events WHERE category = 'person'")
        persons = {"total": 0, "dikenal": 0, "unknown": 0}
        for label, n in conn.execute(sql + " GROUP BY label", params):
            persons["total"] += n
            persons["unknown" if label == "Unknown" else "dikenal"] += n

    for key in ("masuk", "keluar"):
        stats[key]["total"] = sum(stats[key][t] for t in VEHICLE_TYPES)
    stats["di_dalam"] = {
        t: max(0, stats["masuk"][t] - stats["keluar"][t]) for t in VEHICLE_TYPES
    }
    stats["di_dalam"]["total"] = sum(stats["di_dalam"][t] for t in VEHICLE_TYPES)
    stats["manual_masuk"] = manual_masuk
    stats["orang"] = persons
    return stats


# ------------------------------------------------------------------ lines
def get_lines():
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, name, x1, y1, x2, y2, flip FROM lines ORDER BY id"
        ).fetchall()
    return [dict(r) | {"flip": bool(r["flip"])} for r in rows]


def add_line(name, x1, y1, x2, y2, flip=False):
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO lines (name, x1, y1, x2, y2, flip, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, x1, y1, x2, y2, int(flip), _now()),
        )
        return cur.lastrowid


def delete_line(line_id):
    with _db() as conn:
        return conn.execute("DELETE FROM lines WHERE id = ?", (line_id,)).rowcount > 0


def toggle_line_flip(line_id):
    """Balik arah 'masuk' sebuah garis. Mengembalikan nilai flip baru, atau None jika tidak ada."""
    with _db() as conn:
        cur = conn.execute("UPDATE lines SET flip = 1 - flip WHERE id = ?", (line_id,))
        if cur.rowcount == 0:
            return None
        return bool(conn.execute("SELECT flip FROM lines WHERE id = ?", (line_id,)).fetchone()[0])


# ------------------------------------------------------------------ faces
def save_face(name, encoding):
    enc_list = np.asarray(encoding).tolist()
    with _db() as conn:
        conn.execute(
            "INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, json.dumps(enc_list))
        )


def get_known_faces():
    with _db() as conn:
        rows = conn.execute("SELECT name, encoding FROM faces").fetchall()
    names, encodings = [], []
    for row in rows:
        names.append(row["name"])
        encodings.append(np.array(json.loads(row["encoding"])))
    return names, encodings

def list_faces():
    """Satu baris per nama, beserta jumlah contoh (sudut wajah) yang tersimpan."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT name, COUNT(*) AS samples FROM faces GROUP BY name ORDER BY MIN(id)"
        ).fetchall()
    return [{"name": r["name"], "samples": r["samples"]} for r in rows]


def rename_face(old_name, new_name):
    """Ganti nama wajah + riwayat log orang. Jika new_name sudah ada, keduanya otomatis digabung."""
    with _db() as conn:
        cur = conn.execute("UPDATE faces SET name = ? WHERE name = ?", (new_name, old_name))
        if cur.rowcount == 0:
            return False
        conn.execute(
            "UPDATE events SET label = ? WHERE category = 'person' AND label = ?",
            (new_name, old_name),
        )
        return True


def delete_face(name):
    with _db() as conn:
        return conn.execute("DELETE FROM faces WHERE name = ?", (name,)).rowcount > 0


def clear_events(category):
    with _db() as conn:
        return conn.execute("DELETE FROM events WHERE category = ?", (category,)).rowcount
