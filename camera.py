"""Membuka, memindai, dan menguji sumber kamera (USB/webcam lokal maupun kamera IP/RTSP/HTTP)."""
import os

# Paksa RTSP lewat TCP (mencegah gambar pecah/hijau pada CCTV). Harus diset sebelum OpenCV membuka stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import threading
import time

import cv2

from app_settings import build_camera_url, mask_url

IS_WINDOWS = os.name == "nt"
NET_TIMEOUT_MS = 8000  # batas waktu membuka / membaca stream jaringan

_scan_lock = threading.Lock()


def usb_backends(choice="auto"):
    """Daftar backend OpenCV yang dicoba untuk kamera lokal."""
    if not IS_WINDOWS:
        return ((cv2.CAP_ANY, "auto"),)
    if choice == "dshow":
        return ((cv2.CAP_DSHOW, "DirectShow"),)
    if choice == "msmf":
        return ((cv2.CAP_MSMF, "Media Foundation"),)
    return ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"))


def _first_frame(cap, tries, delay):
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(delay)
    return None


def _open_usb(index, backend, width, height, fps, mjpg):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    if mjpg:  # format MJPG biasanya memungkinkan 30 FPS pada webcam
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kurangi delay (diabaikan jika tidak didukung)
    return cap


def _open_network(url):
    cap = None
    try:  # OpenCV >= 4.5.x: batas waktu agar tidak menggantung saat kamera tidak terjangkau
        cap = cv2.VideoCapture(
            url, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NET_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, NET_TIMEOUT_MS],
        )
    except Exception:
        cap = None
    if cap is None:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_source(cam, width=0, height=0, fps=0):
    """Buka sumber kamera dari pengaturan.
    Mengembalikan (cap, frame_pertama, label) atau (None, None, pesan_kesalahan).
    Melempar ValueError jika alamat kamera IP tidak valid."""
    if cam["type"] == "ip":
        url = build_camera_url(cam)
        shown = mask_url(url)
        cap = _open_network(url)
        if cap is None:
            return None, None, (
                f"Tidak bisa terhubung ke {shown}. Periksa alamat, port, username/password, "
                "dan pastikan komputer satu jaringan dengan kamera."
            )
        frame = _first_frame(cap, tries=3, delay=0.5)
        if frame is None:
            cap.release()
            return None, None, f"Terhubung ke {shown} tetapi tidak ada gambar. Cek path stream dan password."
        return cap, frame, f"{shown} ({frame.shape[1]}x{frame.shape[0]})"

    index = int(cam["index"])
    for backend, _name in usb_backends(cam.get("backend", "auto")):
        for mjpg in (True, False):
            cap = _open_usb(index, backend, width, height, fps, mjpg)
            if cap is None:
                continue
            frame = _first_frame(cap, tries=10, delay=0.1)
            if frame is not None:
                return cap, frame, f"Webcam/USB nomor {index} ({frame.shape[1]}x{frame.shape[0]})"
            cap.release()
    return None, None, (
        f"Kamera nomor {index} tidak memberi gambar. Tutup aplikasi lain yang memakai kamera, "
        "periksa izin kamera Windows, atau pilih nomor lain."
    )


def test_source(cam, busy_index=None):
    """Uji koneksi tanpa mengubah pengaturan. busy_index = nomor kamera lokal yang sedang dipakai aplikasi."""
    if cam["type"] == "usb" and busy_index is not None and int(cam["index"]) == busy_index:
        return {"ok": True, "message": "Kamera ini sedang dipakai aplikasi dan berfungsi."}
    try:
        cap, frame, label = open_source(cam)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    if cap is None:
        return {"ok": False, "message": label}
    cap.release()
    return {"ok": True, "message": f"Berhasil: {label}", "width": frame.shape[1], "height": frame.shape[0]}


def scan_usb(max_index=8, busy_index=None, backend_choice="auto"):
    """Cari kamera lokal bernomor 0..max_index. Berhenti setelah 3 nomor berturut-turut kosong.
    Nomor yang sedang dipakai aplikasi ditandai 'busy' tanpa dibuka."""
    results, misses = [], 0
    with _scan_lock:
        for index in range(max_index + 1):
            if index == busy_index:
                results.append({"index": index, "state": "busy"})
                misses = 0
                continue
            found = None
            for backend, name in usb_backends(backend_choice):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    frame = _first_frame(cap, tries=5, delay=0.1)
                    if frame is not None:
                        found = {"index": index, "state": "ok", "backend": name,
                                 "width": frame.shape[1], "height": frame.shape[0]}
                cap.release()
                if found:
                    break
            if found:
                results.append(found)
                misses = 0
            else:
                misses += 1
                if misses >= 3:
                    break
    return results