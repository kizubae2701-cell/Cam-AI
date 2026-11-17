# CCTV AI Analytics System

Sistem analitik CCTV berbasis Artificial Intelligence untuk menghitung jumlah kendaraan (mobil & motor) dan mendeteksi wajah 3secara real-time. Dibangun menggunakan arsitektur backend FastAPI dan algoritma Computer Vision.

## 🚀 Fitur Utama
* **Vehicle Counting (Hitbox):** Menghitung jumlah mobil dan motor yang masuk/keluar melewati garis batas virtual menggunakan model **YOLOv8**.
* **Real-time Face Recognition:** Mendeteksi wajah menggunakan library `face_recognition`. Wajah yang tidak dikenal akan dilabeli abu-abu, sedangkan wajah yang tersimpan di database akan dilabeli hijau dengan namanya.
* **Interactive Web Dashboard:** Antarmuka web responsif untuk memantau stream CCTV dan mendaftarkan wajah baru (klik untuk simpan nama).
* **Database Tracking:** Log otomatis ke database SQLite untuk setiap kendaraan yang lewat dan wajah yang dikenali.

## 🛠️ Teknologi yang Digunakan
* **Backend:** Python 3.11, FastAPI, Uvicorn
* **AI & Computer Vision:** OpenCV, Ultralytics (YOLO), dlib
* **Database:** SQLite3
* **Frontend:** HTML, Vanilla JavaScript, Jinja2

## ⚙️ Persiapan & Instalasi

1. Pastikan Anda menggunakan **Python 3.11**.
2. Buka terminal di folder proyek ini dan jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:
   ```bash
   pip install fastapi uvicorn opencv-python ultralytics face_recognition numpy jinja2 python-multipart
