import os

# Paksa RTSP lewat TCP (mencegah gambar pecah/hijau pada CCTV). Harus sebelum import cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import asyncio
import csv
import io
import platform
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import torch
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO

import database
import settings as app_settings
from plate_reader import PlateReader, normalize_plate
from tracking import PresenceLogger, VehicleTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ konfigurasi
# 0 = webcam laptop, 1/2 = kamera USB, atau URL, mis. "rtsp://user:pass@192.168.1.10:554/stream1"
# Bisa juga lewat environment variable CCTV_URL.
CAMERA_SOURCE = os.environ.get("CCTV_URL", 0)
CAMERA_BACKENDS = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
CAMERA_FPS = 30

# Sisakan CPU untuk thread kamera & encoder JPEG
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# Kelas COCO yang dihitung sebagai kendaraan (bus & truk dihitung sebagai "Mobil")
VEHICLE_CLASSES = {2: "Mobil", 3: "Motor", 5: "Mobil", 7: "Mobil"}

# Kecepatan deteksi (video tetap tampil di FPS kamera; ini hanya mengatur beban deteksi)
YOLO_IMGSZ = 416              # makin kecil makin cepat (kelipatan 32); 320 tercepat, 640 paling akurat
FACE_EVERY_N_CYCLES = 2       # deteksi wajah tiap N siklus deteksi

# Pengenalan wajah
FACE_TOLERANCE = 0.55         # makin kecil makin ketat
PRESENCE_ABSENT_GAP = 60      # detik: orang baru dicatat lagi setelah menghilang selama ini

# Deteksi lokasi wajah (YuNet)
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_SCORE = 0.6              # makin kecil makin sensitif (0.5-0.9), tetapi lebih banyak salah deteksi
FACE_MIN_PX = 20              # wajah lebih kecil dari ini diabaikan
FACE_MAX_WIDTH = 960          # frame lebih lebar diperkecil dulu agar cepat

STATUS_ACTIVE = "Kamera aktif"

COLOR_CAR = (255, 160, 56)    # BGR
COLOR_BIKE = (56, 190, 255)
COLOR_LINE = (80, 80, 255)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def ascii_text(text, fallback=""):
    """cv2.putText hanya mendukung ASCII."""
    return text.encode("ascii", "ignore").decode() or fallback


class Engine:
    """Dua thread latar:
    - kamera : membaca frame, menggambar hasil deteksi terakhir, lalu menyiarkannya (FPS = FPS kamera)
    - deteksi: menjalankan YOLO + wajah pada frame terbaru dengan kecepatannya sendiri
    Dengan begitu video tetap mulus walau deteksi butuh waktu."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.camera_thread = None
        self.detect_thread = None
        self.status = "Memulai..."
        self.detector_status = "Memuat model YOLO..."
        self.latest_jpeg = None
        self.frame_id = 0
        self.frame_size = (640, 480)
        self.fps_video = 0.0
        self.fps_detect = 0.0

        self._raw = None                  # frame mentah terbaru (tidak pernah digambari)
        self._raw_event = threading.Event()
        self.tracks_overlay = []          # hasil deteksi kendaraan terakhir
        self.faces = []                   # [{'box': (top, right, bottom, left), 'enc', 'name'}]
        self.known_names, self.known_encodings = [], []
        self.lines = []

        self.model = None
        self.face_det = None
        self._face_det_size = None
        self.tracker = VehicleTracker()
        self.plates = PlateReader()
        self.presence = PresenceLogger(absent_gap=PRESENCE_ABSENT_GAP)
        self._cycle = 0
        self.settings = app_settings.default_settings()
        self._camera_reconfigure = threading.Event()

    # ----------------------------------------------------------- siklus hidup
    def start(self):
        database.init_db()
        self.settings = app_settings.normalize(database.get_setting("app", None))
        self.reload_faces()
        self.reload_lines()
        self.plates.start()
        self.stop_event.clear()
        self.camera_thread = threading.Thread(target=self.run_camera, daemon=True, name="camera")
        self.detect_thread = threading.Thread(target=self.run_detector, daemon=True, name="detector")
        self.camera_thread.start()
        self.detect_thread.start()

    def shutdown(self):
        self.stop_event.set()
        self._raw_event.set()
        for t in (self.camera_thread, self.detect_thread):
            if t:
                t.join(timeout=5)

    def reload_lines(self):
        self.lines = database.get_lines()

    def reload_faces(self):
        names, encs = database.get_known_faces()
        with self.lock:
            self.known_names, self.known_encodings = names, encs

    # ------------------------------------------------------------------ kamera
    def run_camera(self):
        while not self.stop_event.is_set():
            self.status = "Membuka kamera..."
            cap, frame = self._open_camera()
            if cap is None:
                self.status = (
                    "Kamera tidak memberi frame. Tutup aplikasi lain yang memakai kamera "
                    "dan pastikan izin kamera Windows aktif."
                )
                self.stop_event.wait(3)
                continue
            self.status = STATUS_ACTIVE
            try:
                self._capture_loop(cap, frame)
            finally:
                cap.release()
            if not self.stop_event.is_set():
                self.status = "Kamera terputus, mencoba lagi..."
                self.stop_event.wait(1)

    @staticmethod
    def _try_open(source, backend, mjpg):
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            cap.release()
            return None, None
        if isinstance(source, int):
            if mjpg:  # format MJPG biasanya memungkinkan 30 FPS pada webcam
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kurangi delay (diabaikan jika tidak didukung)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            time.sleep(0.1)
        cap.release()
        return None, None

    def _open_camera(self):
        cam = self.settings["camera"]
        source = cam["index"] if cam["type"] == "usb" else app_settings.build_camera_url(cam)
        if os.name == "nt":
            backends = {"auto": CAMERA_BACKENDS, "dshow": (cv2.CAP_DSHOW,), "msmf": (cv2.CAP_MSMF,)}[cam.get("backend", "auto")]
        else:
            backends = (cv2.CAP_ANY,)
        if isinstance(source, int):
            sources, modes = tuple(dict.fromkeys((source, 1, 2))), (True, False)
        else:
            sources, modes = (source,), (False,)
        for source in sources:
            for backend in backends:
                for mjpg in modes:
                    cap, frame = self._try_open(source, backend, mjpg)
                    if cap is not None:
                        quality = self.settings["quality"]
                        if quality["width"] and quality["height"]:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, quality["width"])
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, quality["height"])
                        if quality["fps"]:
                            cap.set(cv2.CAP_PROP_FPS, quality["fps"])
                        return cap, frame
        return None, None

    @staticmethod
    def _normalize(frame):
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    def _capture_loop(self, cap, frame):
        failures = 0
        last = time.time()
        while not self.stop_event.is_set():
            if self._camera_reconfigure.is_set():
                self._camera_reconfigure.clear()
                return
            if frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        return
                    frame = None
                    time.sleep(0.1)
                    continue
            failures = 0

            frame = self._normalize(frame)
            if frame is None:
                continue
            self.frame_size = (frame.shape[1], frame.shape[0])
            self._raw = frame            # untuk thread deteksi (tidak boleh diubah)
            self._raw_event.set()

            try:
                annotated = self.render(frame.copy())
            except Exception as exc:  # satu frame bermasalah tidak boleh mematikan stream
                print(f"[Engine] Error menggambar frame: {exc}")
                annotated = frame
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
                    self.frame_id += 1

            now = time.time()
            self.fps_video = self._ema(self.fps_video, now - last)
            last = now
            frame = None

    @staticmethod
    def _ema(current, dt):
        inst = 1.0 / max(dt, 1e-3)
        return inst if current == 0 else 0.9 * current + 0.1 * inst

    # ---------------------------------------------------------------- deteksi
    def run_detector(self):
        try:
            self.detector_status = "Memuat model YOLO..."
            self.model = YOLO("yolov8n.pt")  # otomatis diunduh jika belum ada
        except Exception as exc:
            self.detector_status = f"Gagal memuat model YOLO: {exc}"
            return
        self.face_det = self._init_face_detector()
        self.detector_status = "aktif"
        last = time.time()
        while not self.stop_event.is_set():
            if not self._raw_event.wait(0.5):
                continue
            self._raw_event.clear()
            frame = self._raw
            if frame is None:
                continue
            try:
                self.detect(frame, time.time())
            except Exception as exc:
                print(f"[Engine] Error deteksi: {exc}")
            now = time.time()
            self.fps_detect = self._ema(self.fps_detect, now - last)
            last = now

    def detect(self, frame, now):
        """Deteksi kendaraan (+ lintasan garis) dan wajah pada satu frame. Frame tidak diubah."""
        h, w = frame.shape[:2]
        self._cycle += 1

        quality = self.settings["quality"]
        kwargs = {"classes": list(VEHICLE_CLASSES), "imgsz": quality["yolo_imgsz"], "conf": app_settings.VEHICLE_CONF[quality["vehicle_sens"]], "verbose": False}
        if self.settings["device"] != "auto":
            kwargs["device"] = "0" if self.settings["device"] == "gpu" and torch.cuda.is_available() else "cpu"
        results = self.model(frame, **kwargs)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[int(box.cls[0])]))

        tracks, crossings = self.tracker.update(detections, self.lines, (w, h), now)
        for c in crossings:
            event_id = database.add_event(
                "vehicle",
                c["label"],
                direction=c["direction"],
                plate=database.NO_PLATE,
                line_name=c["line"]["name"],
                source="auto",
            )
            x1, y1, x2, y2 = c["bbox"]
            crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)].copy()
            self.plates.submit(event_id, crop)
        self.tracks_overlay = tracks

        if self._cycle % quality["face_every"] == 0:
            self._detect_faces(frame, now)

    # ---------------------------------------------------------------- wajah
    def _init_face_detector(self):
        """YuNet: detektor wajah ringan yang kuat untuk wajah samping/tertutup/kecil."""
        if not os.path.exists(FACE_MODEL_PATH):
            try:
                import urllib.request

                print("[Engine] Mengunduh model wajah YuNet...")
                urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
            except Exception as exc:
                print(f"[Engine] Gagal mengunduh YuNet ({exc}). Memakai detektor HOG sebagai cadangan.")
                return None
        try:
            return cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (320, 320), FACE_SCORE, 0.3, 5000)
        except Exception as exc:
            print(f"[Engine] YuNet tidak bisa dimuat ({exc}). Memakai detektor HOG sebagai cadangan.")
            return None

    def _locate_faces(self, frame):
        """Kembalikan daftar kotak wajah (top, right, bottom, left) dalam piksel frame asli."""
        h, w = frame.shape[:2]
        if self.face_det is not None:
            scale = min(1.0, FACE_MAX_WIDTH / w)
            img = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
            ih, iw = img.shape[:2]
            if self._face_det_size != (iw, ih):
                self.face_det.setInputSize((iw, ih))
                self._face_det_size = (iw, ih)
            _, res = self.face_det.detect(img)
            boxes = []
            if res is not None:
                for r in res:
                    x, y, bw, bh = (float(v) / scale for v in r[:4])
                    left, top = max(0, int(x)), max(0, int(y))
                    right, bottom = min(w, int(x + bw)), min(h, int(y + bh))
                    if right - left < FACE_MIN_PX or bottom - top < FACE_MIN_PX:
                        continue
                    boxes.append((top, right, bottom, left))
            return boxes

        # Cadangan bila YuNet gagal dimuat: HOG pada 50% ukuran + upsample
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=2)
        return [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in locs]

    def _detect_faces(self, frame, now):
        locations = self._locate_faces(frame)
        if not locations:
            with self.lock:
                self.faces = []
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        try:
            encodings = face_recognition.face_encodings(rgb, locations)
        except RuntimeError as exc:
            print(f"[Engine] Deteksi wajah gagal: {exc}")
            return

        faces = []
        for box, enc in zip(locations, encodings):
            name = self._identify(enc)
            faces.append({"box": box, "enc": enc, "name": name})
            # Dicatat sekali per kehadiran, bukan tiap terlihat
            if self.presence.observe(name, enc, now):
                database.add_event("person", name, source="auto")
        with self.lock:
            self.faces = faces

    def _identify(self, encoding):
        with self.lock:
            names, encs = list(self.known_names), list(self.known_encodings)
        if not encs:
            return "Unknown"
        dist = face_recognition.face_distance(encs, encoding)
        best = int(np.argmin(dist))
        tolerance = app_settings.FACE_TOLERANCE[self.settings["quality"]["match_level"]]
        return names[best] if dist[best] <= tolerance else "Unknown"

    # ---------------------------------------------------------------- gambar
    def render(self, frame):
        """Gambar hasil deteksi terakhir ke frame (dipanggil di FPS kamera)."""
        view = self.settings.get("view", "semua")
        for t in self.tracks_overlay if view in ("parkir", "semua") else []:
            x1, y1, x2, y2 = t["bbox"]
            color = COLOR_CAR if t["label"] == "Mobil" else COLOR_BIKE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{t['label']} #{t['id']}", (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for ln in self.lines if view in ("parkir", "semua") else []:
            self._draw_line(frame, ln)
        with self.lock:
            faces = list(self.faces)
        for f in faces if view in ("wajah", "semua") else []:
            top, right, bottom, left = f["box"]
            color = (0, 255, 0) if f["name"] != "Unknown" else (160, 160, 160)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, ascii_text(f["name"], "?"), (left, max(14, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    @staticmethod
    def _draw_line(frame, ln):
        h, w = frame.shape[:2]
        a = (int(ln["x1"] * w), int(ln["y1"] * h))
        b = (int(ln["x2"] * w), int(ln["y2"] * h))
        cv2.line(frame, a, b, COLOR_LINE, 2)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / length, dx / length  # normal menuju sisi "masuk"
        if ln.get("flip"):
            nx, ny = -nx, -ny
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        tip = (int(mid[0] + nx * 34), int(mid[1] + ny * 34))
        cv2.arrowedLine(frame, mid, tip, COLOR_LINE, 2, tipLength=0.4)
        cv2.putText(frame, "MASUK", (tip[0] + 4, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LINE, 1)
        cv2.putText(frame, ascii_text(ln["name"], "Garis"), (a[0] + 4, max(14, a[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 2)

    # ------------------------------------------------------------ interaksi web
    def find_face(self, nx, ny):
        w, h = self.frame_size
        px, py = nx * w, ny * h
        with self.lock:
            faces = list(self.faces)
        for f in faces:
            top, right, bottom, left = f["box"]
            if left <= px <= right and top <= py <= bottom:
                return f
        return None

    def register_face(self, face, name):
        database.save_face(name, face["enc"])
        with self.lock:
            self.known_names.append(name)
            self.known_encodings.append(face["enc"])
            face["name"] = name


engine = Engine()


@asynccontextmanager
async def lifespan(_app):
    engine.start()
    yield
    engine.shutdown()


app = FastAPI(lifespan=lifespan)


def _settings_choices():
    return {
        "windows": os.name == "nt", "resolutions": app_settings.RESOLUTIONS,
        "fps": app_settings.FPS_CHOICES, "yolo": app_settings.YOLO_SIZES,
        "face_every": app_settings.FACE_INTERVALS,
    }


def _settings_response():
    current = app_settings.normalize(database.get_setting("app", None))
    return {"settings": app_settings.public_view(current), "choices": _settings_choices(),
            "presets": app_settings.PRESETS}


@app.get("/api/settings")
def api_get_settings():
    return _settings_response()


@app.post("/api/settings")
def api_save_settings(body: dict):
    current = app_settings.normalize(database.get_setting("app", None))
    try:
        updated = app_settings.apply_patch(current, body)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    database.set_setting("app", updated)
    # Apply live detector parameters; camera source is picked up on the next reconnect.
    engine.settings = updated
    if current["camera"] != updated["camera"]:
        engine._camera_reconfigure.set()
    return _settings_response()


@app.get("/api/camera/scan")
def api_camera_scan(backend: str = "auto"):
    if backend not in app_settings.BACKENDS:
        raise HTTPException(400, "Metode pembacaan tidak dikenal.")
    backends = {"auto": CAMERA_BACKENDS, "dshow": (cv2.CAP_DSHOW,), "msmf": (cv2.CAP_MSMF,)} if os.name == "nt" else {"auto": (cv2.CAP_ANY,), "dshow": (cv2.CAP_ANY,), "msmf": (cv2.CAP_ANY,)}
    devices = []
    for index in range(6):
        cap = None
        try:
            for api in backends[backend]:
                cap = cv2.VideoCapture(index, api)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        devices.append({"index": index, "state": "ready", "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
                        break
                    devices.append({"index": index, "state": "busy", "width": 0, "height": 0})
                    break
                cap.release(); cap = None
        finally:
            if cap is not None:
                cap.release()
    return {"devices": devices}


@app.post("/api/camera/test")
def api_camera_test(body: dict):
    current = app_settings.normalize(database.get_setting("app", None))
    try:
        cam = app_settings.preview_camera(current["camera"], body.get("camera", {}))
        source = cam["index"] if cam["type"] == "usb" else app_settings.build_camera_url(cam)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    backend = {"auto": CAMERA_BACKENDS, "dshow": (cv2.CAP_DSHOW,), "msmf": (cv2.CAP_MSMF,)}.get(cam.get("backend"), CAMERA_BACKENDS) if os.name == "nt" else (cv2.CAP_ANY,)
    cap = None
    try:
        for api in backend:
            attempt = cv2.VideoCapture(source, api)
            if attempt.isOpened():
                cap = attempt; break
            attempt.release()
        ok, frame = cap.read() if cap is not None else (False, None)
        return {"ok": bool(ok and frame is not None), "message": "Kamera berhasil dihubungkan." if ok and frame is not None else "Kamera tidak dapat dibuka atau tidak mengirim frame."}
    finally:
        if cap is not None:
            cap.release()


@app.get("/api/system")
def api_system():
    cuda = bool(torch.cuda.is_available())
    conf = app_settings.normalize(database.get_setting("app", None))
    active = "GPU" if conf["device"] == "gpu" and cuda else "CPU"
    return {"device_active": active, "cuda": cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else "",
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda or "", "dlib_cuda": bool(getattr(face_recognition.api, "dlib", None) and getattr(face_recognition.api.dlib, "DLIB_USE_CUDA", False)),
            "face_detector": "YuNet" if os.path.exists(FACE_MODEL_PATH) else "HOG", "cpu_count": os.cpu_count() or 1, "opencv": cv2.__version__}


# ------------------------------------------------------------------ halaman & video
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


async def _mjpeg():
    last = -1
    while not engine.stop_event.is_set():
        with engine.lock:
            jpeg, frame_id = engine.latest_jpeg, engine.frame_id
        if jpeg is None or frame_id == last:
            await asyncio.sleep(0.01)
            continue
        last = frame_id
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def api_status():
    return {
        "camera": engine.status,
        "active": engine.status == STATUS_ACTIVE,
        "detector": engine.detector_status,
        "fps_video": round(engine.fps_video, 1),
        "fps_detect": round(engine.fps_detect, 1),
        "plate_ocr": engine.plates.state,
        "plate_ocr_detail": engine.plates.detail,
        "frame": {"w": engine.frame_size[0], "h": engine.frame_size[1]},
    }


# ------------------------------------------------------------------ statistik & log
def _valid_date(value):
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Format tanggal harus YYYY-MM-DD")
    return value


def _filters(category, vehicle_type, direction, search, date_from, date_to):
    return {
        "category": category,
        "vehicle_type": vehicle_type,
        "direction": direction,
        "search": (search or "").strip()[:100],
        "date_from": _valid_date(date_from),
        "date_to": _valid_date(date_to),
    }


@app.get("/api/stats")
def api_stats(scope: str = Query("today", pattern="^(today|all)$")):
    return database.get_stats(scope)


@app.get("/api/events")
def api_events(
    category: str = "all",          # all | vehicle | person
    vehicle_type: str = "all",      # all | Mobil | Motor
    direction: str = "all",         # all | masuk | keluar
    search: str = "",
    date_from: str = "",
    date_to: str = "",
    sort: str = Query("newest", pattern="^(newest|oldest)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    filters = _filters(category, vehicle_type, direction, search, date_from, date_to)
    rows, total = database.query_events(sort=sort, limit=limit, offset=offset, **filters)
    return {"events": rows, "total": total}


def _csv_safe(value):
    """Cegah formula injection saat CSV dibuka di Excel."""
    value = "" if value is None else str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@") else value


@app.get("/api/events.csv")
def api_events_csv(
    category: str = "all",
    vehicle_type: str = "all",
    direction: str = "all",
    search: str = "",
    date_from: str = "",
    date_to: str = "",
    sort: str = Query("newest", pattern="^(newest|oldest)$"),
):
    filters = _filters(category, vehicle_type, direction, search, date_from, date_to)
    rows, _total = database.query_events(sort=sort, limit=100000, offset=0, **filters)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Tanggal", "Jam", "Tipe", "Jenis/Nama", "Arah", "Plat Nomor", "Garis", "Sumber", "Jumlah", "Catatan"])
    for e in rows:
        date, clock = e["ts"].split(" ")
        writer.writerow([
            date, clock,
            "Kendaraan" if e["category"] == "vehicle" else "Orang",
            _csv_safe(e["label"]), e["direction"] or "",
            _csv_safe(e["plate"]), _csv_safe(e["line_name"]),
            "Manual" if e["source"] == "manual" else "Otomatis",
            e["count"], _csv_safe(e["note"]),
        ])
    filename = f"log_cctv_{datetime.now():%Y%m%d_%H%M%S}.csv"
    return Response(
        ("\ufeff" + out.getvalue()).encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class PlateIn(BaseModel):
    plate: str = Field("", max_length=20)


@app.patch("/api/events/{event_id}/plate")
def api_set_plate(event_id: int, body: PlateIn):
    cleaned = re.sub(r"[^A-Z0-9 ]", "", body.plate.upper()).strip()
    plate = normalize_plate(cleaned) or re.sub(r"\s+", " ", cleaned) or database.NO_PLATE
    if not database.update_event_plate(event_id, plate):
        raise HTTPException(404, "Log kendaraan tidak ditemukan")
    return {"status": "success", "plate": plate}


@app.delete("/api/events/{event_id}")
def api_delete_event(event_id: int):
    if not database.delete_event(event_id):
        raise HTTPException(404, "Log tidak ditemukan")
    return {"status": "success"}


class ClearIn(BaseModel):
    category: str


@app.post("/api/events/clear")
def api_clear_events(body: ClearIn):
    """Hapus semua log pada satu kategori (person | vehicle)."""
    if body.category not in ("person", "vehicle"):
        raise HTTPException(400, "Kategori harus person atau vehicle")
    return {"status": "success", "deleted": database.clear_events(body.category)}


class ManualIn(BaseModel):
    vehicle_type: str
    direction: str
    count: int = Field(1, ge=1, le=9999)
    plate: str = Field("", max_length=20)
    note: str = Field("", max_length=200)


@app.post("/api/manual")
def api_manual(body: ManualIn):
    if body.vehicle_type not in database.VEHICLE_TYPES:
        raise HTTPException(400, "Jenis kendaraan harus Mobil atau Motor")
    if body.direction not in database.DIRECTIONS:
        raise HTTPException(400, "Arah harus masuk atau keluar")
    cleaned = re.sub(r"[^A-Z0-9 ]", "", body.plate.upper()).strip()
    plate = normalize_plate(cleaned) or re.sub(r"\s+", " ", cleaned) or database.NO_PLATE
    event_id = database.add_event(
        "vehicle", body.vehicle_type, direction=body.direction, plate=plate,
        count=body.count, source="manual", note=body.note.strip() or None,
    )
    return {"status": "success", "id": event_id}


# ------------------------------------------------------------------ garis virtual
class LineIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


@app.get("/api/lines")
def api_lines():
    return {"lines": database.get_lines()}


@app.post("/api/lines")
def api_add_line(body: LineIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Nama garis wajib diisi")
    if ((body.x2 - body.x1) ** 2 + (body.y2 - body.y1) ** 2) ** 0.5 < 0.03:
        raise HTTPException(400, "Garis terlalu pendek")
    line_id = database.add_line(name, body.x1, body.y1, body.x2, body.y2)
    engine.reload_lines()
    return {"status": "success", "id": line_id}


@app.post("/api/lines/{line_id}/flip")
def api_flip_line(line_id: int):
    flip = database.toggle_line_flip(line_id)
    if flip is None:
        raise HTTPException(404, "Garis tidak ditemukan")
    engine.reload_lines()
    return {"status": "success", "flip": flip}


@app.delete("/api/lines/{line_id}")
def api_delete_line(line_id: int):
    if not database.delete_line(line_id):
        raise HTTPException(404, "Garis tidak ditemukan")
    engine.reload_lines()
    return {"status": "success"}


# ------------------------------------------------------------------ wajah terdaftar
def _clean_person_name(raw):
    name = (raw or "").strip()[:60]
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong")
    if name.lower() == "unknown":
        raise HTTPException(400, "Nama 'Unknown' dicadangkan untuk wajah yang belum dikenal")
    return name


@app.get("/api/faces")
def api_faces():
    return {"faces": database.list_faces()}


class FaceRenameIn(BaseModel):
    name: str
    new_name: str


@app.post("/api/faces/rename")
def api_rename_face(body: FaceRenameIn):
    new_name = _clean_person_name(body.new_name)
    if not database.rename_face(body.name, new_name):
        raise HTTPException(404, "Wajah tidak ditemukan")
    engine.reload_faces()
    engine.presence.reset()
    return {"status": "success", "name": new_name}


class FaceDeleteIn(BaseModel):
    name: str


@app.post("/api/faces/delete")
def api_delete_face(body: FaceDeleteIn):
    if not database.delete_face(body.name):
        raise HTTPException(404, "Wajah tidak ditemukan")
    engine.reload_faces()
    engine.presence.reset()
    return {"status": "success"}


@app.get("/api/face_at")
def api_face_at(x: float = Query(..., ge=0, le=1), y: float = Query(..., ge=0, le=1)):
    face = engine.find_face(x, y)
    return {"found": face is not None, "name": face["name"] if face else None}


@app.post("/register_face")
async def register_face(x: float = Form(...), y: float = Form(...), name: str = Form(...)):
    """x, y = posisi klik ternormalisasi (0..1) pada video."""
    try:
        name = _clean_person_name(name)
    except HTTPException as exc:
        return JSONResponse({"status": "error", "message": exc.detail})
    face = engine.find_face(x, y)
    if face is None:
        return JSONResponse({"status": "error", "message": "Gagal. Anda tidak mengklik tepat di area kotak wajah."})
    engine.register_face(face, name)
    return JSONResponse({"status": "success", "message": f"Wajah '{name}' berhasil disimpan ke database!"})
