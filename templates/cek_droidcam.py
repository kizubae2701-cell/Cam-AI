"""Diagnosis koneksi DroidCam (Wi-Fi), langkah demi langkah, terpisah dari aplikasi CCTV.

Cara pakai (venv aktif, server aplikasi dihentikan dulu, aplikasi DroidCam di HP terbuka):
    python cek_droidcam.py 192.168.45.175 4747
Tanpa argumen, dipakai IP dan port bawaan di bawah.
"""
import socket
import sys
import urllib.request

import cv2

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.45.175"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 4747


def langkah(teks):
    print(f"\n{teks}")


# ---------------------------------------------------------------- 1. jaringan
langkah(f"1) Menghubungi HP di {IP}:{PORT} ...")
try:
    socket.create_connection((IP, PORT), timeout=5).close()
    print("   OK: HP terjangkau dan port terbuka.")
except Exception as exc:
    print(f"   GAGAL: {exc}")
    print("   Artinya laptop tidak bisa menjangkau HP. Periksa:")
    print("   - aplikasi DroidCam di HP sedang terbuka (layar menyala) dan IP/port-nya sama dengan yang diketik")
    print("   - HP dan laptop memakai Wi-Fi yang sama (IP laptop dari 'ipconfig' sebaiknya berawalan sama)")
    print("   - Wi-Fi tidak memakai 'isolasi klien/AP isolation' (sering aktif di Wi-Fi tamu/hotspot)")
    print("   - DroidCam Client / browser yang sedang memakai DroidCam ditutup")
    sys.exit(1)

# ------------------------------------------------------------ 2. alamat video
langkah("2) Mencari alamat video yang benar ...")
kandidat = []
for path in ("/video", "/mjpegfeed", "/"):
    url = f"http://{IP}:{PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            ctype = resp.headers.get("Content-Type", "")
            print(f"   {path:11s} -> HTTP {resp.status}, {ctype}")
            if any(k in ctype.lower() for k in ("multipart", "image", "video")):
                kandidat.append(path)
    except Exception as exc:
        print(f"   {path:11s} -> gagal: {exc}")

if not kandidat:
    print("   Tidak ada alamat yang mengirim video. Kemungkinan ada koneksi lain yang sedang memakai DroidCam")
    print("   (tutup DroidCam Client dan tab browser), atau versi DroidCam memakai path lain.")
    sys.exit(1)

# -------------------------------------------------------------------- 3. OpenCV
langkah("3) Membuka dengan OpenCV (cara yang dipakai aplikasi) ...")
berhasil = None
for path in kandidat:
    url = f"http://{IP}:{PORT}{path}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    frame = None
    if cap.isOpened():
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            frame = None
    cap.release()
    if frame is not None:
        print(f"   {path}: BERHASIL, gambar {frame.shape[1]}x{frame.shape[0]}")
        berhasil = path
        break
    print(f"   {path}: OpenCV tidak mendapat gambar")

langkah("HASIL")
if berhasil:
    print("Isi di dashboard (Pengaturan > Kamera > Kamera IP):")
    print(f"   Alamat       : {IP}")
    print("   Protokol     : http")
    print(f"   Port         : {PORT}")
    print(f"   Path stream  : {berhasil}")
    print("   Username/Password: kosong")
else:
    print("Jaringan dan alamat video ada, tetapi OpenCV gagal membacanya.")
    print("Coba: perbarui OpenCV (pip install -U opencv-python), tutup koneksi lain ke DroidCam, lalu ulangi.")