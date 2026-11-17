"""Pengaturan aplikasi: nilai bawaan, preset kualitas, validasi, dan penyusun alamat kamera.

Semua pengaturan disimpan sebagai satu dict (JSON di tabel `settings`):
  camera  : sumber kamera (usb / ip)
  quality : resolusi, FPS, kecepatan & sensitivitas deteksi
  device  : auto | cpu | gpu
  view    : parkir | wajah | semua | bersih
"""
import copy
import os
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

# ------------------------------------------------------------------ pilihan yang diizinkan
RESOLUTIONS = [(0, 0), (320, 240), (640, 480), (1280, 720), (1920, 1080)]  # (0, 0) = asli kamera
FPS_CHOICES = [0, 10, 15, 20, 25, 30, 60]                                   # 0 = maksimal
YOLO_SIZES = [320, 416, 480, 640]
FACE_INTERVALS = [1, 2, 3, 4, 6]
VIEWS = ("parkir", "wajah", "semua", "bersih")
DEVICES = ("auto", "cpu", "gpu")
BACKENDS = ("auto", "dshow", "msmf")
CAMERA_TYPES = ("usb", "ip")
SCHEMES = ("rtsp", "rtsps", "http", "https", "rtmp")
PRESET_NAMES = ("auto", "hemat", "seimbang", "akurat", "kustom")

# Tingkat sensitivitas 1..10 -> nilai ambang. Makin tinggi = makin sensitif.
VEHICLE_CONF = {1: 0.65, 2: 0.55, 3: 0.45, 4: 0.40, 5: 0.35, 6: 0.30, 7: 0.25, 8: 0.20, 9: 0.15, 10: 0.10}
FACE_SCORE = {1: 0.90, 2: 0.85, 3: 0.80, 4: 0.75, 5: 0.70, 6: 0.65, 7: 0.60, 8: 0.55, 9: 0.50, 10: 0.40}
# Kelonggaran pengenalan nama 1..10: makin tinggi = makin mudah dikenali, tapi risiko salah nama naik.
FACE_TOLERANCE = {1: 0.40, 2: 0.43, 3: 0.46, 4: 0.49, 5: 0.52, 6: 0.55, 7: 0.58, 8: 0.61, 9: 0.64, 10: 0.67}

PRESETS = {
    "hemat": dict(width=640, height=480, fps=15, yolo_imgsz=320, face_every=3, vehicle_sens=6, face_sens=6),
    "seimbang": dict(width=640, height=480, fps=30, yolo_imgsz=416, face_every=2, vehicle_sens=7, face_sens=7),
    "akurat": dict(width=1280, height=720, fps=30, yolo_imgsz=640, face_every=1, vehicle_sens=8, face_sens=8),
    # auto: kecepatan deteksi (yolo_imgsz & face_every) diatur sendiri oleh program sesuai beban
    "auto": dict(width=640, height=480, fps=30, yolo_imgsz=416, face_every=2, vehicle_sens=7, face_sens=7),
}


def default_settings():
    camera = {
        "type": "usb", "index": 0, "backend": "auto",
        "scheme": "rtsp", "address": "", "port": 0, "path": "",
        "username": "", "password": "",
    }
    env_url = os.environ.get("CCTV_URL")  # kompatibel dengan versi lama
    if env_url:
        camera.update(type="ip", address=env_url)
        try:
            camera = _clean_camera(camera, {})
        except ValueError:
            camera.update(type="usb", address="")
    quality = dict(PRESETS["seimbang"], preset="seimbang", match_level=6)
    return {"camera": camera, "quality": quality, "device": "auto", "view": "semua"}


# ------------------------------------------------------------------ validasi
def _int(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} tidak valid.")
    if value in ("", None):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"{label} harus berupa angka.")


def _int_in(value, lo, hi, label):
    number = _int(value, label)
    if not lo <= number <= hi:
        raise ValueError(f"{label} harus antara {lo} dan {hi}.")
    return number


def _text(value, limit):
    return str(value if value is not None else "").strip()[:limit]


def _split_credentials(address):
    """Pisahkan user:password dari link lengkap agar tidak tersimpan/terkirim di alamat."""
    if "://" not in address:
        return address, None, None
    try:
        parts = urlsplit(address)
        port = parts.port
    except ValueError:
        raise ValueError("Alamat atau port kamera tidak valid.")
    if parts.username is None:
        return address, None, None
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if port:
        host += f":{port}"
    clean = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    password = unquote(parts.password) if parts.password is not None else ""
    return clean, unquote(parts.username), password


def _clean_camera(current, patch):
    if not isinstance(patch, dict):
        raise ValueError("Format pengaturan kamera tidak valid.")
    cam = dict(current)
    if "type" in patch:
        if patch["type"] not in CAMERA_TYPES:
            raise ValueError("Jenis kamera tidak dikenal.")
        cam["type"] = patch["type"]
    if "index" in patch:
        cam["index"] = _int_in(patch["index"], 0, 20, "Nomor kamera")
    if "backend" in patch:
        if patch["backend"] not in BACKENDS:
            raise ValueError("Metode pembacaan tidak dikenal.")
        cam["backend"] = patch["backend"]
    if "scheme" in patch:
        if patch["scheme"] not in SCHEMES:
            raise ValueError("Protokol tidak dikenal.")
        cam["scheme"] = patch["scheme"]
    if "username" in patch:
        cam["username"] = _text(patch["username"], 100)
    if "address" in patch:  # link lengkap berisi user:password menimpa kolom username/password
        address, user, password = _split_credentials(_text(patch["address"], 300))
        cam["address"] = address
        if user is not None:
            cam["username"], cam["password"] = _text(user, 100), password[:200]
    if "port" in patch:
        cam["port"] = _int_in(patch["port"], 0, 65535, "Port")
    if "path" in patch:
        cam["path"] = _text(patch["path"], 300)
    if patch.get("clear_password"):
        cam["password"] = ""
    elif patch.get("password"):  # kosong = pertahankan yang tersimpan
        cam["password"] = str(patch["password"])[:200]
    if cam["type"] == "ip":
        build_camera_url(cam)  # memvalidasi; ValueError jika tidak valid
    return cam


def _clean_quality(current, patch):
    if not isinstance(patch, dict):
        raise ValueError("Format pengaturan kualitas tidak valid.")
    q = dict(current)
    preset = patch.get("preset", q.get("preset"))
    if preset not in PRESET_NAMES:
        raise ValueError("Preset tidak dikenal.")
    if "preset" in patch and preset in PRESETS:
        q.update(PRESETS[preset])
    q["preset"] = preset
    if "width" in patch or "height" in patch:
        pair = (_int(patch.get("width", q["width"]), "Lebar"), _int(patch.get("height", q["height"]), "Tinggi"))
        if pair not in RESOLUTIONS:
            raise ValueError("Resolusi tidak tersedia.")
        q["width"], q["height"] = pair
    for key, allowed, label in (
        ("fps", FPS_CHOICES, "FPS"),
        ("yolo_imgsz", YOLO_SIZES, "Ukuran proses kendaraan"),
        ("face_every", FACE_INTERVALS, "Interval deteksi wajah"),
    ):
        if key in patch:
            value = _int(patch[key], label)
            if value not in allowed:
                raise ValueError(f"{label} tidak tersedia.")
            q[key] = value
    for key, label in (("vehicle_sens", "Sensitivitas kendaraan"), ("face_sens", "Sensitivitas wajah"),
                       ("match_level", "Kelonggaran pengenalan nama")):
        if key in patch:
            q[key] = _int_in(patch[key], 1, 10, label)
    return q


def apply_patch(current, patch):
    """Gabungkan perubahan ke pengaturan; ValueError (pesan bahasa Indonesia) jika tidak valid."""
    if not isinstance(patch, dict):
        raise ValueError("Format pengaturan tidak valid.")
    new = copy.deepcopy(current)
    if "camera" in patch:
        new["camera"] = _clean_camera(new["camera"], patch["camera"])
    if "quality" in patch:
        new["quality"] = _clean_quality(new["quality"], patch["quality"])
    if "device" in patch:
        if patch["device"] not in DEVICES:
            raise ValueError("Perangkat pemrosesan tidak dikenal.")
        new["device"] = patch["device"]
    if "view" in patch:
        if patch["view"] not in VIEWS:
            raise ValueError("Mode tampilan tidak dikenal.")
        new["view"] = patch["view"]
    return new


def preview_camera(current, patch):
    """Pengaturan kamera hasil gabungan (untuk uji koneksi tanpa menyimpan)."""
    return _clean_camera(current, patch)


def normalize(raw):
    """Muat dari database; bagian yang rusak/tidak valid diganti nilai bawaan."""
    current = default_settings()
    if not isinstance(raw, dict):
        return current
    for key in ("camera", "quality", "device", "view"):
        if key in raw:
            try:
                current = apply_patch(current, {key: raw[key]})
            except ValueError:
                pass
    return current


# ------------------------------------------------------------------ alamat kamera
def build_camera_url(cam):
    """Susun URL lengkap dari kolom alamat/port/path/username/password. ValueError jika tidak valid."""
    address = (cam.get("address") or "").strip()
    if not address:
        raise ValueError("Alamat kamera belum diisi.")
    if "://" not in address:
        scheme = cam.get("scheme") or "rtsp"
        path = (cam.get("path") or "").strip()
        if path and not path.startswith("/"):
            path = "/" + path
        port = int(cam.get("port") or 0)
        host = address.rstrip("/")
        if port and ":" not in host:
            host = f"{host}:{port}"
        address = f"{scheme}://{host}{path}"
    try:
        parts = urlsplit(address)
        port = parts.port
    except ValueError:
        raise ValueError("Alamat atau port kamera tidak valid.")
    scheme = parts.scheme.lower()
    if scheme not in SCHEMES:
        raise ValueError("Alamat harus diawali rtsp://, http://, https://, rtsps:// atau rtmp://.")
    host = parts.hostname
    if not host:
        raise ValueError("Alamat kamera tidak valid.")
    user, password = parts.username, parts.password
    if user is None and cam.get("username"):
        user = quote(cam["username"], safe="")
        password = quote(cam["password"], safe="") if cam.get("password") else None
    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc += f":{port}"
    if user is not None:
        netloc = f"{user}:{password}@{netloc}" if password is not None else f"{user}@{netloc}"
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


_SECRET_QUERY = re.compile(r"(?i)((?:pass(?:word)?|pwd)=)[^&]*")


def mask_url(url):
    """URL yang aman ditampilkan (password disamarkan)."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "(alamat tidak valid)"
    query = _SECRET_QUERY.sub(r"\1***", parts.query)
    if parts.username is None:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if port:
        host += f":{port}"
    cred = parts.username + (":***" if parts.password is not None else "")
    return urlunsplit((parts.scheme, f"{cred}@{host}", parts.path, query, parts.fragment))


def describe_camera(cam):
    if cam["type"] == "usb":
        return f"Webcam/USB nomor {cam['index']}"
    try:
        return mask_url(build_camera_url(cam))
    except ValueError as exc:
        return str(exc)


def public_view(settings):
    """Salinan pengaturan untuk browser: password tidak pernah dikirim."""
    out = copy.deepcopy(settings)
    cam = out["camera"]
    cam["password_set"] = bool(cam.pop("password", ""))
    cam["display"] = describe_camera(settings["camera"])
    return out