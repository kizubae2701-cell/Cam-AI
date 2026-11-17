"""Pelacak kendaraan + wajah (memberi #ID sementara), deteksi lintasan garis (masuk / keluar),
dan pencatat kehadiran orang (sekali per kehadiran).

Garis disimpan sebagai koordinat ternormalisasi (0..1) dan memiliki arah:
sisi "negatif" -> sisi "positif" dihitung MASUK, sebaliknya KELUAR
(dibalik jika flag `flip` aktif). Panah di video menunjuk ke arah MASUK.

Catatan: #ID adalah nomor pelacakan, bukan identitas. Objek yang hilang dari kamera cukup lama
(`max_age`) akan mendapat ID baru saat muncul lagi.
"""
import math
import threading
import time
from collections import Counter, deque

import numpy as np

CROSS_COOLDOWN = 3.0  # detik: kendaraan yang sama tidak dihitung dua kali di garis yang sama


def side_of_line(a, b, p):
    """Tanda >= 0 berarti satu sisi garis A->B, < 0 sisi lainnya."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def segments_intersect(p0, p1, a, b):
    """Apakah ruas gerak p0->p1 memotong ruas garis a->b."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (p1[0] - p0[0], p1[1] - p0[1])
    denom = _cross(r[0], r[1], s[0], s[1])
    if denom == 0:
        return False
    qp = (p0[0] - a[0], p0[1] - a[1])
    t = _cross(qp[0], qp[1], s[0], s[1]) / denom  # posisi di sepanjang garis
    u = _cross(qp[0], qp[1], r[0], r[1]) / denom  # posisi di sepanjang gerakan
    return 0 <= t <= 1 and 0 <= u <= 1


def crossing_direction(p0, p1, a, b, flip=False):
    """'masuk' / 'keluar' jika gerakan p0->p1 melintasi garis a->b, selain itu None."""
    if not segments_intersect(p0, p1, a, b):
        return None
    s0 = side_of_line(a, b, p0)
    s1 = side_of_line(a, b, p1)
    if (s0 < 0) == (s1 < 0):
        return None
    entering = s0 < 0
    if flip:
        entering = not entering
    return "masuk" if entering else "keluar"


def _match(detections_centers, limits, tracks, now):
    """Pasangan greedy terdekat antara deteksi dan track. Mengembalikan {indeks_deteksi: id_track}."""
    pairs = []
    for di, center in enumerate(detections_centers):
        for tid, tr in tracks.items():
            d = math.hypot(center[0] - tr.center[0], center[1] - tr.center[1])
            if d <= limits[di]:
                pairs.append((d, di, tid))
    pairs.sort()
    matched_det, matched_track = {}, set()
    for _d, di, tid in pairs:
        if di in matched_det or tid in matched_track:
            continue
        matched_det[di] = tid
        matched_track.add(tid)
    return matched_det


# ================================================================== kendaraan
class _Track:
    __slots__ = ("id", "center", "bbox", "last_seen", "votes", "crossed")

    def __init__(self, track_id, center, bbox, label, now):
        self.id = track_id
        self.center = center
        self.bbox = bbox
        self.last_seen = now
        self.votes = Counter({label: 1})
        self.crossed = {}  # line_id -> waktu lintasan terakhir

    @property
    def label(self):
        return self.votes.most_common(1)[0][0]


class VehicleTracker:
    def __init__(self, max_age=1.5, min_dist=60.0, size_factor=0.8):
        self.max_age = max_age
        self.min_dist = min_dist
        self.size_factor = size_factor
        self.tracks = {}
        self._next_id = 1

    def set_next_id(self, number):
        """Lanjutkan penomoran (mis. dari ID terbesar di database) agar tidak mengulang."""
        self._next_id = max(self._next_id, int(number))

    def update(self, detections, lines, frame_size, now=None):
        """detections: [((x1, y1, x2, y2), label), ...] dalam piksel.
        lines: [{'id', 'name', 'x1', 'y1', 'x2', 'y2', 'flip'}, ...] ternormalisasi.
        Mengembalikan (track_aktif, lintasan)."""
        now = time.time() if now is None else now
        w, h = frame_size

        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > self.max_age]:
            del self.tracks[tid]

        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, _ in detections]
        limits = [
            max(self.min_dist, self.size_factor * max(b[2] - b[0], b[3] - b[1])) for b, _ in detections
        ]
        matched_det = _match(centers, limits, self.tracks, now)

        px_lines = [
            (ln, (ln["x1"] * w, ln["y1"] * h), (ln["x2"] * w, ln["y2"] * h)) for ln in lines
        ]

        active, crossings = [], []
        for di, (bbox, label) in enumerate(detections):
            center = centers[di]
            if di in matched_det:
                tr = self.tracks[matched_det[di]]
                prev = tr.center
                tr.center, tr.bbox, tr.last_seen = center, bbox, now
                tr.votes[label] += 1
                for ln, a, b in px_lines:
                    if now - tr.crossed.get(ln["id"], -1e9) < CROSS_COOLDOWN:
                        continue
                    direction = crossing_direction(prev, center, a, b, ln.get("flip", False))
                    if direction:
                        tr.crossed[ln["id"]] = now
                        crossings.append(
                            {
                                "track_id": tr.id,
                                "label": tr.label,
                                "direction": direction,
                                "line": ln,
                                "bbox": bbox,
                            }
                        )
            else:
                tr = _Track(self._next_id, center, bbox, label, now)
                self._next_id += 1
                self.tracks[tr.id] = tr
            active.append({"id": tr.id, "bbox": bbox, "label": tr.label})
        return active, crossings


# ===================================================================== wajah
class _FaceTrack:
    __slots__ = ("id", "center", "last_seen", "names")

    def __init__(self, track_id, center, now, window):
        self.id = track_id
        self.center = center
        self.last_seen = now
        self.names = deque(maxlen=window)

    def vote(self, name):
        """Nama stabil dari beberapa pengamatan terakhir (mencegah label berkedip
        antara nama dan Unknown). Seri -> nama dikenal yang paling baru menang."""
        self.names.append(name)
        counts = Counter(self.names)
        top = max(counts.values())
        known = [n for n, c in counts.items() if c == top and n != "Unknown"]
        if known:
            for n in reversed(self.names):
                if n in known:
                    return n
        return "Unknown"

    def set_name(self, name):
        self.names.clear()
        self.names.extend([name] * 3)


class FaceTracker:
    """Memberi #ID pada setiap wajah yang terdeteksi dan mengikutinya antar siklus deteksi."""

    def __init__(self, max_age=2.5, min_dist=40.0, size_factor=1.0, vote_window=5):
        self.max_age = max_age
        self.min_dist = min_dist
        self.size_factor = size_factor
        self.vote_window = vote_window
        self.tracks = {}
        self._next_id = 1

    def set_next_id(self, number):
        self._next_id = max(self._next_id, int(number))

    def reset(self):
        self.tracks.clear()

    def force_name(self, track_id, name):
        tr = self.tracks.get(track_id)
        if tr:
            tr.set_name(name)

    def update(self, boxes, now=None):
        """boxes: [(top, right, bottom, left), ...]. Mengembalikan daftar track sejajar dengan boxes."""
        now = time.time() if now is None else now
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > self.max_age]:
            del self.tracks[tid]

        centers = [((l + r) / 2.0, (t + b) / 2.0) for (t, r, b, l) in boxes]
        limits = [max(self.min_dist, self.size_factor * max(r - l, b - t)) for (t, r, b, l) in boxes]
        matched = _match(centers, limits, self.tracks, now)

        out = []
        for bi, center in enumerate(centers):
            if bi in matched:
                tr = self.tracks[matched[bi]]
                tr.center, tr.last_seen = center, now
            else:
                tr = _FaceTrack(self._next_id, center, now, self.vote_window)
                self._next_id += 1
                self.tracks[tr.id] = tr
            out.append(tr)
        return out


class PresenceLogger:
    """Mencatat orang SEKALI per kehadiran, bukan setiap terlihat.

    - Wajah dikenal: dicatat saat pertama muncul; baru dicatat lagi jika
      menghilang lebih dari `absent_gap` detik.
    - Unknown: dibedakan antar-orang lewat kemiripan encoding, dan baru dicatat
      setelah terlihat `confirm_hits` kali (mencegah kedipan deteksi
      "dikenal -> Unknown" membuat log palsu).
    """

    def __init__(self, absent_gap=60.0, unknown_tolerance=0.6, confirm_hits=3, confirm_window=5.0):
        self.absent_gap = absent_gap
        self.unknown_tolerance = unknown_tolerance
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self._lock = threading.Lock()
        self._known = {}     # nama -> waktu terakhir terlihat
        self._unknown = []   # [{'enc', 'last', 'hits', 'logged'}]

    def reset(self):
        with self._lock:
            self._known.clear()
            self._unknown.clear()

    def observe(self, name, encoding, now=None):
        """True jika kehadiran ini harus dicatat ke log."""
        now = time.time() if now is None else now
        with self._lock:
            if name != "Unknown":
                last = self._known.get(name)
                self._known[name] = now
                return last is None or now - last > self.absent_gap
            return self._observe_unknown(encoding, now)

    def _observe_unknown(self, enc, now):
        enc = np.asarray(enc)
        # buang entri yang sudah lama hilang / tidak pernah terkonfirmasi
        self._unknown = [
            u for u in self._unknown
            if now - u["last"] <= (self.absent_gap if u["logged"] else self.confirm_window)
        ]
        best, best_d = None, self.unknown_tolerance
        for u in self._unknown:
            d = float(np.linalg.norm(u["enc"] - enc))
            if d <= best_d:
                best, best_d = u, d
        if best is None:
            best = {"enc": enc, "last": now, "hits": 0, "logged": False}
            self._unknown.append(best)
        best["last"], best["enc"] = now, enc
        best["hits"] += 1
        if not best["logged"] and best["hits"] >= self.confirm_hits:
            best["logged"] = True
            return True
        return False