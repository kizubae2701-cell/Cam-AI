"""Pembaca plat nomor (OCR) yang berjalan di thread terpisah agar video tidak tersendat.

Butuh paket opsional `easyocr` (pip install easyocr). Tanpa itu, plat tetap dicatat sebagai '???'.
"""
import importlib.util
import queue
import re
import threading

import cv2

import database

_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "


def _fix(chars, want_digit):
    """Ubah huruf <-> angka yang sering tertukar oleh OCR. Mengembalikan (teks, biaya) atau None."""
    out, cost = "", 0
    for ch in chars:
        if want_digit:
            if ch.isdigit():
                out += ch
            elif ch in _TO_DIGIT:
                out += _TO_DIGIT[ch]
                cost += 1
            else:
                return None
        else:
            if ch.isalpha():
                out += ch
            elif ch in _TO_LETTER:
                out += _TO_LETTER[ch]
                cost += 1
            else:
                return None
    return out, cost


def normalize_plate(text):
    """Cocokkan teks OCR ke format plat Indonesia (1-2 huruf, 1-4 angka, 0-3 huruf).
    Mengembalikan mis. 'B 1234 ABC' atau None jika tidak mirip plat."""
    s = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    if not 3 <= len(s) <= 9:
        return None
    best = None
    for p in (1, 2):
        for q in (0, 1, 2, 3):
            mid_len = len(s) - p - q
            if not 1 <= mid_len <= 4:
                continue
            if not any(ch.isdigit() for ch in s[p : p + mid_len]):
                continue  # plat selalu punya angka asli; cegah kata biasa (mis. "HELLO") lolos
            parts = (
                _fix(s[:p], want_digit=False),
                _fix(s[p : p + mid_len], want_digit=True),
                _fix(s[p + mid_len :], want_digit=False),
            )
            if any(part is None for part in parts):
                continue
            cost = sum(part[1] for part in parts)
            candidate = " ".join(part[0] for part in parts if part[0])
            if best is None or cost < best[0]:
                best = (cost, candidate)
    if best and best[0] <= 1:
        return best[1]
    return None


class PlateReader:
    def __init__(self):
        self.state = "missing" if importlib.util.find_spec("easyocr") is None else "idle"
        self.detail = "easyocr belum terpasang di interpreter aplikasi" if self.state == "missing" else ""
        self._queue = queue.Queue(maxsize=8)
        self._reader = None
        self._thread = None

    @property
    def available(self):
        return self.state in ("idle", "loading", "ready")

    def start(self):
        if self.state != "idle":
            if self.state == "missing":
                print("[OCR] easyocr belum terpasang -> plat dicatat '???'. Pasang dengan: pip install easyocr")
            return
        self.state = "loading"
        self._thread = threading.Thread(target=self._worker, daemon=True, name="plate-ocr")
        self._thread.start()

    def submit(self, event_id, crop):
        if not self.available or crop is None or crop.size == 0:
            return
        try:
            self._queue.put_nowait((event_id, crop))
        except queue.Full:
            pass  # antrean penuh: lewati, plat tetap '???'

    def _worker(self):
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:  # model gagal diunduh / dimuat
            self.detail = str(exc)
            print(f"[OCR] Gagal memuat easyocr: {exc}")
            self.state = "error"
            return
        self.state = "ready"
        self.detail = ""
        while True:
            event_id, crop = self._queue.get()
            try:
                plate = self.read(crop)
                if plate:
                    database.update_event_plate(event_id, plate)
            except Exception as exc:
                print(f"[OCR] Error membaca plat: {exc}")

    def read(self, crop):
        h = crop.shape[0]
        roi = crop[int(h * 0.4) :, :]  # plat ada di bagian bawah kendaraan
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texts = self._reader.readtext(gray, detail=0, allowlist=_ALLOWLIST)
        candidates = [normalize_plate("".join(texts))] + [normalize_plate(t) for t in texts]
        candidates = [c for c in candidates if c]
        return max(candidates, key=len) if candidates else None
