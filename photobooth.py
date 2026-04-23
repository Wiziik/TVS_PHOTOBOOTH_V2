#!/usr/bin/env python3
"""
TVS Photobooth — single-file edition
- Fullscreen live preview (pygame, kmsdrm/fbcon/x11 auto-detect)
- Arduino Leonardo USB-MIDI toggle → countdown → capture → ESC/POS print
- AV-to-USB capture card via OpenCV (auto-detects working device index)
- ESC/POS thermal receipt printer USB 0x0525:0xA700 (80mm / 576 dots)
- Non-blocking print (background thread — no UI freeze)
- Scanline + vignette CRT overlay, minimal_ui mode
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import logging
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import mido
import pygame

from web_uploader import WebUploader, generate_ticket_code

from tvstore_receipt import print_receipt


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_NAME       = "PiPhotobooth"
PHOTOS_DIR     = Path("./photos")
PRINT_QUEUE_DIR = Path("./prints_queue")
DEFAULT_INI_PATH = Path("./photobooth.ini")


# ---------------------------------------------------------------------------
# Y2K UI theme
# ---------------------------------------------------------------------------

Y2K = {
    "panel_fill":        (10,  12,  18,  150),
    "panel_fill_2":      (18,  20,  30,  165),
    "panel_stroke":      (140, 190, 255, 180),
    "panel_stroke_hot":  (255, 70,  200, 200),
    "panel_stroke_ok":   (60,  255, 200, 200),
    "panel_stroke_warn": (255, 170, 60,  210),
    "panel_stroke_err":  (255, 80,  80,  220),
    "text":              (235, 245, 255),
    "text_dim":          (190, 205, 220),
    "accent_cyan":       (60,  255, 220),
    "accent_magenta":    (255, 70,  200),
    "accent_lime":       (160, 255, 90),
    "scanline_alpha":    18,
}


def draw_bevel_panel(
    target: pygame.Surface,
    rect: pygame.Rect,
    fill_rgba: tuple,
    stroke_rgba: tuple,
    radius: int,
    stroke: int,
    bevel: int = 2,
) -> None:
    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(panel, fill_rgba, panel.get_rect(), border_radius=radius)
    top_h = max(2, rect.h // 3)
    gloss = pygame.Surface((rect.w, top_h), pygame.SRCALPHA)
    pygame.draw.rect(gloss, (255, 255, 255, 24), gloss.get_rect(), border_radius=radius)
    panel.blit(gloss, (0, 0))
    pygame.draw.rect(panel, (255, 255, 255, 32), panel.get_rect(), width=bevel, border_radius=radius)
    inset = panel.get_rect().inflate(-bevel * 2, -bevel * 2)
    pygame.draw.rect(panel, (0, 0, 0, 40), inset, width=bevel, border_radius=max(0, radius - bevel))
    pygame.draw.rect(panel, stroke_rgba, panel.get_rect(), width=stroke, border_radius=radius)
    target.blit(panel, rect.topleft)


def draw_glow_text(
    target: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    pos: tuple,
    color: tuple,
    glow_color: tuple,
    *,
    center: bool = False,
    glow_px: int = 5,
    glow_layers: int = 6,
    alpha_max: int = 170,
) -> pygame.Rect:
    base = font.render(text, True, color)
    rect = base.get_rect(center=pos) if center else base.get_rect(topleft=pos)
    for i in range(glow_layers, 0, -1):
        a = int(alpha_max * (i / glow_layers) ** 2)
        glow = font.render(text, True, glow_color).convert_alpha()
        glow.set_alpha(a)
        off = int((i / glow_layers) * glow_px)
        for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off),
                       (-off, -off), (-off, off), (off, -off), (off, off)):
            target.blit(glow, rect.move(dx, dy))
    target.blit(base, rect)
    return rect


def make_scanlines(size: tuple, alpha: int = 18, spacing: int = 3) -> pygame.Surface:
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    line = pygame.Surface((w, 1), pygame.SRCALPHA)
    line.fill((0, 0, 0, alpha))
    for y in range(0, h, spacing):
        surf.blit(line, (0, y))
    return surf


def make_vignette(size: tuple, strength: int = 55) -> pygame.Surface:
    w, h = size
    v = pygame.Surface((w, h), pygame.SRCALPHA)
    pad = max(12, min(w, h) // 18)
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, 0, w, pad))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, h - pad, w, pad))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, 0, pad, h))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(w - pad, 0, pad, h))
    return v


# ---------------------------------------------------------------------------
# Display init
# ---------------------------------------------------------------------------

def init_pygame_display(fullscreen: bool) -> pygame.Surface:
    """Try kmsdrm (card0, card1), fbcon, wayland, x11 in order."""
    flags = pygame.FULLSCREEN if fullscreen else 0

    # If caller already set SDL_VIDEODRIVER, honour it first
    if os.environ.get("SDL_VIDEODRIVER"):
        try:
            return pygame.display.set_mode((0, 0), flags)
        except pygame.error as e:
            logging.warning("pygame set_mode failed for driver=%s: %s",
                            os.environ["SDL_VIDEODRIVER"], e)

    candidates = [
        ("kmsdrm", {"SDL_VIDEO_KMSDRM_DEVICE": "/dev/dri/card1"}),
        ("kmsdrm", {"SDL_VIDEO_KMSDRM_DEVICE": "/dev/dri/card0"}),
        ("fbcon",  {"SDL_FBDEV": "/dev/fb0"}),
        ("wayland", {}),
        ("x11",    {"DISPLAY": os.environ.get("DISPLAY", ":0")}),
    ]

    for drv, extra_env in candidates:
        try:
            os.environ["SDL_VIDEODRIVER"] = drv
            for k, v in extra_env.items():
                os.environ[k] = v
            pygame.display.quit()
            pygame.display.init()
            surf = pygame.display.set_mode((0, 0), flags)
            logging.info("Display driver: %s (%s)", drv, extra_env)
            return surf
        except pygame.error as e:
            logging.warning("pygame set_mode failed for %s %s: %s", drv, extra_env, e)

    raise pygame.error("No usable SDL video driver found.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_dt() -> datetime:
    return datetime.now()

def today_folder_name(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def timestamp_name(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def ensure_dirs() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    PRINT_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

def safe_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def safe_load_sound(path: str) -> Optional[pygame.mixer.Sound]:
    try:
        if not path:
            return None
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        if not p.exists():
            return None
        return pygame.mixer.Sound(str(p))
    except Exception:
        logging.exception("Failed to load sound: %s", path)
        return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MidiMapping:
    start_note:    int = 60   # Middle C — matches Leonardo_MIDI_Note.ino
    filter_note:   int = 38
    print_note:    int = 40
    delete_note:   int = 41
    brightness_cc: int = 1
    countdown_cc:  int = 2


@dataclass
class CameraConfig:
    picamera2_preview_size: Tuple[int, int] = (1280, 720)
    picamera2_still_size:   Tuple[int, int] = (3280, 2464)
    picamera2_swap_rb:      bool            = False
    opencv_device_index:    int             = -1   # -1 = auto-scan
    opencv_width:           int             = 1280
    opencv_height:          int             = 720


@dataclass
class AppConfig:
    midi_port_name:         str            = ""
    midi_mapping:           MidiMapping    = dataclasses.field(default_factory=MidiMapping)
    fullscreen:             bool           = True
    rotate_preview_degrees: int            = 0
    countdown_seconds:      int            = 3
    last_photo_show_seconds: int           = 3
    brightness:             float          = 1.0
    filter_cycle:           Tuple[str,...] = ("none", "bw", "retro")
    default_filter:         str            = "none"
    copies:                 int            = 1
    image_reduce_factor:    float          = 1.0
    qr_url:                 str            = ""
    print_brightness:       float          = 1.3
    print_contrast:         float          = 1.1
    idle_music:             str            = ""
    minimal_ui:             bool           = True
    countdown_wav:          str            = "./countdown.wav"
    shutter_wav:            str            = "./shutter.wav"
    pay_overlay_image:        str          = "./logo.png"
    pay_overlay_price_text:   str          = "1€ = 3 PHOTOS"
    pay_overlay_footer_text:  str          = "PHOTOBOOTH // ANALOG DREAMS"
    payment_accepted_text:    str          = "PAIEMENT ACCEPTÉ"
    payment_accepted_seconds: float        = 1.8
    ready_text:               str          = "PRÊT"
    credits_per_unlock:       int          = 1
    photos_per_session:       int          = 3
    between_shots_seconds:    float        = 2.5
    # Web upload (QR landing page + email delivery)
    upload_url:               str          = ""
    upload_token:             str          = ""
    ticket_url_template:      str          = "https://photos.tvstore.fr/t/{code}"
    camera:                   CameraConfig = dataclasses.field(default_factory=CameraConfig)


def _bool(v, default):
    if v is None: return default
    s = str(v).strip().lower()
    if s in ("1","true","yes","y","on"):  return True
    if s in ("0","false","no","n","off"): return False
    return default

def _int(v, default):
    try:    return int(str(v).strip())
    except: return default

def _float(v, default):
    try:    return float(str(v).strip())
    except: return default


DEFAULT_INI = """\
[app]
fullscreen = true
minimal_ui = true
rotate_preview_degrees = 0
countdown_seconds = 3
last_photo_show_seconds = 3
brightness = 1.0
countdown_wav = ./countdown.wav
shutter_wav = ./shutter.wav
; Music played on loop when idle. Leave empty to disable.
idle_music =
; Overlay shown any time there are zero credits. Image path empty = no logo.
pay_overlay_image = ./logo.png
pay_overlay_price_text = 1€ = 3 PHOTOS
pay_overlay_footer_text = PHOTOBOOTH // ANALOG DREAMS
; Transition message shown for a moment after a successful /unlock.
payment_accepted_text = PAIEMENT ACCEPTÉ
payment_accepted_seconds = 1.8
; One credit = one session. A session takes photos_per_session shots and prints them as a strip.
credits_per_unlock = 1
photos_per_session = 3
; Pause between captures within a session (seconds). Shows ready_text while waiting.
between_shots_seconds = 2.5
ready_text = PRÊT

[filters]
cycle = none,bw,retro
default = none

[midi]
; leave empty to auto-pick (skips "Midi Through")
port_name =
start_note = 60
filter_note = 38
print_note = 40
delete_note = 41
brightness_cc = 1
countdown_cc = 2

[printer]
copies = 1
; 1.0 = full print width, 0.8 = 20% narrower (less print data)
image_reduce_factor = 1.0
; QR URL override from INI. Leave empty to use receipt_text.txt QR_URL
qr_url =
; Photo brightness before printing (1.0 = no change, >1 = lighter, <1 = darker)
print_brightness = 1.3
; Photo contrast before printing (1.0 = no change, >1 = more contrast)
print_contrast = 1.1

[camera]
; -1 = auto-scan device indices until one works
opencv_device_index = -1
opencv_width = 1280
opencv_height = 720
picamera2_preview_width = 1280
picamera2_preview_height = 720
picamera2_still_width = 3280
picamera2_still_height = 2464
picamera2_swap_rb = false

[web]
; Leave upload_url / upload_token empty to disable web upload (QR falls back to printer.qr_url).
; Secrets — keep this section out of your public git push.
upload_url =
upload_token =
ticket_url_template = https://photos.tvstore.fr/t/{code}
"""


def load_config(path: Path = DEFAULT_INI_PATH) -> AppConfig:
    if not path.exists():
        path.write_text(DEFAULT_INI, encoding="utf-8")
        logging.info("Created default config: %s", path)

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding="utf-8")

    app  = cp["app"]      if cp.has_section("app")     else {}
    filt = cp["filters"]  if cp.has_section("filters") else {}
    midi = cp["midi"]     if cp.has_section("midi")    else {}
    prn  = cp["printer"]  if cp.has_section("printer") else {}
    cam  = cp["camera"]   if cp.has_section("camera")  else {}
    web  = cp["web"]      if cp.has_section("web")     else {}

    cycle_raw    = str(filt.get("cycle", "none,bw,retro"))
    filter_cycle = tuple(x.strip() for x in cycle_raw.split(",") if x.strip()) or ("none",)
    default_filter = str(filt.get("default", filter_cycle[0])).strip() or filter_cycle[0]

    mm = MidiMapping(
        start_note    = _int(midi.get("start_note"),    60),
        filter_note   = _int(midi.get("filter_note"),   38),
        print_note    = _int(midi.get("print_note"),    40),
        delete_note   = _int(midi.get("delete_note"),   41),
        brightness_cc = _int(midi.get("brightness_cc"), 1),
        countdown_cc  = _int(midi.get("countdown_cc"),  2),
    )

    cam_cfg = CameraConfig(
        picamera2_preview_size = (_int(cam.get("picamera2_preview_width"), 1280),
                                  _int(cam.get("picamera2_preview_height"), 720)),
        picamera2_still_size   = (_int(cam.get("picamera2_still_width"), 3280),
                                  _int(cam.get("picamera2_still_height"), 2464)),
        picamera2_swap_rb      = _bool(cam.get("picamera2_swap_rb"), False),
        opencv_device_index    = _int(cam.get("opencv_device_index"), -1),
        opencv_width           = _int(cam.get("opencv_width"), 1280),
        opencv_height          = _int(cam.get("opencv_height"), 720),
    )

    return AppConfig(
        midi_port_name          = str(midi.get("port_name", "")).strip(),
        midi_mapping            = mm,
        fullscreen              = _bool(app.get("fullscreen"), True),
        rotate_preview_degrees  = _int(app.get("rotate_preview_degrees"), 0),
        countdown_seconds       = _int(app.get("countdown_seconds"), 3),
        last_photo_show_seconds = _int(app.get("last_photo_show_seconds"), 3),
        brightness              = _float(app.get("brightness"), 1.0),
        filter_cycle            = filter_cycle,
        default_filter          = default_filter,
        copies                  = _int(prn.get("copies"), 1),
        image_reduce_factor     = clamp(_float(prn.get("image_reduce_factor"), 1.0), 0.1, 1.0),
        qr_url                  = str(prn.get("qr_url", "")).strip(),
        print_brightness        = clamp(_float(prn.get("print_brightness"), 1.3), 0.1, 3.0),
        print_contrast          = clamp(_float(prn.get("print_contrast"),   1.1), 0.1, 3.0),
        minimal_ui              = _bool(app.get("minimal_ui"), True),
        countdown_wav           = str(app.get("countdown_wav", "./countdown.wav")).strip(),
        shutter_wav             = str(app.get("shutter_wav",   "./shutter.wav")).strip(),
        idle_music              = str(app.get("idle_music", "")).strip(),
        pay_overlay_image        = str(app.get("pay_overlay_image", "./logo.png")).strip(),
        pay_overlay_price_text   = str(app.get("pay_overlay_price_text", "1€ = 3 PHOTOS")).strip(),
        pay_overlay_footer_text  = str(app.get("pay_overlay_footer_text", "PHOTOBOOTH // ANALOG DREAMS")).strip(),
        payment_accepted_text    = str(app.get("payment_accepted_text", "PAIEMENT ACCEPTÉ")).strip(),
        payment_accepted_seconds = _float(app.get("payment_accepted_seconds"), 1.8),
        ready_text               = str(app.get("ready_text", "PRÊT")).strip(),
        credits_per_unlock       = max(1, _int(app.get("credits_per_unlock"), 1)),
        photos_per_session       = max(1, _int(app.get("photos_per_session"), 3)),
        between_shots_seconds    = max(0.0, _float(app.get("between_shots_seconds"), 2.5)),
        upload_url               = str(web.get("upload_url", "")).strip(),
        upload_token             = str(web.get("upload_token", "")).strip(),
        ticket_url_template      = str(web.get("ticket_url_template", "https://photos.tvstore.fr/t/{code}")).strip(),
        camera                   = cam_cfg,
    )


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------

@dataclass
class MidiEvent:
    type:     str
    note:     Optional[int] = None
    velocity: Optional[int] = None
    control:  Optional[int] = None
    value:    Optional[int] = None
    port:     Optional[str] = None
    raw:      Optional[dict] = None


class MidiController:
    def __init__(self, port_name: str, out_queue: "queue.Queue[MidiEvent]",
                 poll_interval: float = 2.0) -> None:
        self.port_name     = port_name
        self.out_queue     = out_queue
        self.poll_interval = poll_interval
        self._stop         = threading.Event()
        self._thread       = threading.Thread(target=self._run, name="MIDI", daemon=True)
        self._connected    : Optional[str] = None

    @staticmethod
    def list_ports() -> list:
        try:    return mido.get_input_names()
        except: return []

    def start(self) -> None: self._thread.start()
    def stop(self)  -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def connected_port(self) -> Optional[str]: return self._connected

    def _pick_port(self, names: list) -> Optional[str]:
        if not names:
            return None
        if self.port_name:
            needle = self.port_name.lower()
            for n in names:
                if needle in n.lower():
                    return n
            return None
        non_through = [n for n in names if "through" not in n.lower()]
        return (non_through or names)[0]

    def _normalize(self, msg: mido.Message, port: str) -> Optional[MidiEvent]:
        d = msg.dict()
        t = d.get("type")
        if t == "note_on" and int(d.get("velocity", 0)) > 0:
            return MidiEvent(type="note_on", note=int(d["note"]),
                             velocity=int(d["velocity"]), port=port, raw=d)
        if t == "control_change":
            return MidiEvent(type="control_change", control=int(d["control"]),
                             value=int(d["value"]), port=port, raw=d)
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chosen = self._pick_port(self.list_ports())
                if not chosen:
                    self._connected = None
                    time.sleep(self.poll_interval)
                    continue
                logging.info("MIDI connected: %s", chosen)
                with mido.open_input(chosen) as port:
                    self._connected = chosen
                    for msg in port:
                        if self._stop.is_set():
                            break
                        ev = self._normalize(msg, chosen)
                        if ev:
                            self.out_queue.put(ev)
                logging.warning("MIDI disconnected: %s", chosen)
                self._connected = None
            except Exception as e:
                logging.exception("MIDI error: %s", e)
                self._connected = None
                time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class CameraController:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg     = cfg
        self.backend = "none"
        self._picam2 = None
        self._preview_config = None
        self._still_config   = None
        self._opencv_cap     = None
        self._init()

    def _init(self) -> None:
        # Try Picamera2 first (without FrameRate control — UVC cards don't support it)
        try:
            from picamera2 import Picamera2  # type: ignore
            pw, ph = self.cfg.picamera2_preview_size
            sw, sh = self.cfg.picamera2_still_size
            cam = Picamera2()
            self._preview_config = cam.create_preview_configuration(
                main={"size": (pw, ph), "format": "RGB888"},
            )
            self._still_config = cam.create_still_configuration(
                main={"size": (sw, sh), "format": "RGB888"},
            )
            cam.configure(self._preview_config)
            cam.start()
            self._picam2 = cam
            self.backend = "picamera2"
            logging.info("Camera: Picamera2/libcamera")
            return
        except Exception as e:
            logging.warning("Picamera2 failed: %s", e)

        # OpenCV fallback — auto-scan indices
        try:
            import cv2  # type: ignore
            indices = (
                [self.cfg.opencv_device_index]
                if self.cfg.opencv_device_index >= 0
                else list(range(5))
            )
            for idx in indices:
                # Explicit V4L2 backend avoids the "not a capture device" issue
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.opencv_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.opencv_height)
                ok, _ = cap.read()
                if ok:
                    self._opencv_cap = cap
                    self.backend = "opencv"
                    logging.info("Camera: OpenCV V4L2 index=%d", idx)
                    return
                cap.release()
        except Exception as e:
            logging.error("Camera init failed (real backend): %s", e)

        # Mock backend for headless/no-hardware testing
        self.backend = "mock"
        logging.info("Camera: MOCK mode (placeholder frames)")

    def available(self) -> bool:
        return self.backend in ("picamera2", "opencv", "mock")

    def retry(self) -> None:
        self.close()
        time.sleep(0.4)
        self._init()

    def close(self) -> None:
        try:
            if self._picam2:
                self._picam2.stop()
                self._picam2.close()
        except Exception:
            pass
        self._picam2 = self._preview_config = self._still_config = None
        try:
            if self._opencv_cap is not None:
                self._opencv_cap.release()
        except Exception:
            pass
        self._opencv_cap = None
        if self.backend != "mock":
            self.backend = "none"

    @staticmethod
    def _to_rgb(arr: np.ndarray) -> np.ndarray:
        """Normalise any Picamera2 pixel format to uint8 H×W×3 RGB."""
        import cv2  # type: ignore
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr[:, :, :3]                           # XBGR/RGBA → drop alpha
        if arr.ndim == 3 and arr.shape[2] == 2:
            return cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_YUYV)  # YUYV → RGB
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=2)             # greyscale → RGB
        return arr                                         # assume RGB888 already

    def _get_mock_frame(self) -> np.ndarray:
        w, h = 1280, 720
        arr = np.random.randint(40, 60, (h, w, 3), dtype=np.uint8)
        # Add some "vibe"
        ts = time.time()
        c1 = int(127 + 127 * math.sin(ts))
        c2 = int(127 + 127 * math.cos(ts))
        arr[h//2-50:h//2+50, w//2-50:w//2+50] = [c1, c2, 180]
        return arr

    def get_preview_frame(self) -> Optional[np.ndarray]:
        try:
            if self.backend == "picamera2" and self._picam2:
                arr = self._to_rgb(self._picam2.capture_array("main"))
                if self.cfg.picamera2_swap_rb:
                    arr = arr[..., ::-1]
                return arr
            if self.backend == "opencv" and self._opencv_cap is not None:
                import cv2  # type: ignore
                ok, frame = self._opencv_cap.read()
                if not ok:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.backend == "mock":
                return self._get_mock_frame()
        except Exception:
            return None

    def capture_still(self) -> Optional[Image.Image]:
        try:
            if self.backend == "picamera2" and self._picam2:
                self._picam2.stop()
                self._picam2.configure(self._still_config)
                self._picam2.start()
                arr = self._to_rgb(self._picam2.capture_array("main"))
                if self.cfg.picamera2_swap_rb:
                    arr = arr[..., ::-1]
                self._picam2.stop()
                self._picam2.configure(self._preview_config)
                self._picam2.start()
                return Image.fromarray(arr)
            if self.backend == "opencv" and self._opencv_cap is not None:
                arr = self.get_preview_frame()
                return Image.fromarray(arr, mode="RGB") if arr is not None else None
            if self.backend == "mock":
                return Image.fromarray(self._get_mock_frame(), mode="RGB")
        except Exception as e:
            logging.exception("Capture failed: %s", e)
            return None

    def capture_still_and_stop(self) -> Optional[Image.Image]:
        """Capture a frame from the live preview stream, then stop the camera.

        For UVC capture cards, switching to still_config causes the first
        returned frame to be black (camera hasn't stabilised after the mode
        switch).  Grabbing from the already-running preview gets a stable
        frame.  The camera is then fully stopped so its USB isochronous
        transfers don't compete with the printer on the shared bus.
        """
        try:
            if self.backend == "picamera2" and self._picam2:
                # Capture from the already-running preview stream.
                # Do NOT stop/reconfigure/restart: for UVC capture cards the
                # first frame after a mode switch is always black (the sensor
                # hasn't stabilised yet), and the stop→start USB cycle disturbs
                # other devices on the shared bus (printer gets stale stall).
                arr = self._to_rgb(self._picam2.capture_array("main"))
                if self.cfg.picamera2_swap_rb:
                    arr = arr[..., ::-1]
                # Stop and close — do NOT restart preview
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception:
                    pass
                self._picam2 = None
                self.backend = "none"
                return Image.fromarray(arr)
            if self.backend == "opencv" and self._opencv_cap is not None:
                arr = self.get_preview_frame()
                try:
                    self._opencv_cap.release()
                except Exception:
                    pass
                self._opencv_cap = None
                self.backend = "none"
                return Image.fromarray(arr, mode="RGB") if arr is not None else None
        except Exception as e:
            logging.exception("Capture failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def apply_filter(img: Image.Image, filter_name: str, brightness: float) -> Image.Image:
    out = img.convert("RGB")
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if filter_name == "bw":
        return ImageOps.grayscale(out).convert("RGB")
    if filter_name == "retro":
        out = ImageEnhance.Color(out).enhance(0.75)
        out = ImageEnhance.Contrast(out).enhance(1.15)
        overlay = Image.new("RGB", out.size, (255, 230, 200))
        return Image.blend(out, overlay, alpha=0.12)
    return out


# ---------------------------------------------------------------------------
# Print manager — non-blocking
# ---------------------------------------------------------------------------

class PrintManager:
    """Runs ESC/POS printing in a background thread so the UI never freezes."""

    def __init__(
        self,
        copies: int = 1,
        image_reduce_factor: float = 1.0,
        qr_url: str = "",
        print_brightness: float = 1.3,
        print_contrast: float = 1.1,
    ) -> None:
        self.copies   = max(1, copies)
        self.image_reduce_factor = clamp(float(image_reduce_factor), 0.1, 1.0)
        self.qr_url = str(qr_url).strip()
        self.print_brightness = clamp(float(print_brightness), 0.1, 3.0)
        self.print_contrast   = clamp(float(print_contrast),   0.1, 3.0)
        self._lock    = threading.Lock()
        self._printing = False

    def is_printing(self) -> bool:
        with self._lock:
            return self._printing

    def print_async(self, photo_paths,
                    on_done: Optional[callable] = None,
                    qr_url: Optional[str] = None) -> None:
        """Queue a print job. Accepts a Path/str or an iterable of them.
        Silently skips if already printing.
        qr_url overrides the default (e.g. per-session ticket URL)."""
        if isinstance(photo_paths, (str, Path)):
            paths = [Path(photo_paths)]
        else:
            paths = [Path(p) for p in photo_paths]
        effective_qr = qr_url if qr_url is not None else self.qr_url

        with self._lock:
            if self._printing:
                logging.warning("Print already in progress, skipping.")
                return
            self._printing = True

        def _worker():
            try:
                for p in paths:
                    if not p.exists():
                        raise FileNotFoundError(f"Photo not found: {p}")
                    size = p.stat().st_size
                    if size < 5_000:
                        raise ValueError(
                            f"Photo file too small ({p.name}: {size} B) — capture may have failed"
                        )
                logging.warning("[PRINT] Job: %d photo(s) — %s",
                                len(paths), ", ".join(p.name for p in paths))
                for i in range(self.copies):
                    if i > 0:
                        time.sleep(15.0)  # cool-down between copies
                    print_receipt(
                        [str(p) for p in paths],
                        reduce_factor=self.image_reduce_factor,
                        qr_url=effective_qr,
                        brightness=self.print_brightness,
                        contrast=self.print_contrast,
                    )
                msg = f"Printed {self.copies}x OK"
                logging.info(msg)
                if on_done:
                    on_done(True, msg)
            except Exception as e:
                msg = f"Print failed: {e}"
                logging.error(msg)
                # Fallback: copy each to the print queue for later retry
                try:
                    PRINT_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
                    for p in paths:
                        shutil.copy2(p, PRINT_QUEUE_DIR / p.name)
                    msg += f" (queued: {len(paths)})"
                except Exception:
                    pass
                if on_done:
                    on_done(False, msg)
            finally:
                with self._lock:
                    self._printing = False

        threading.Thread(target=_worker, name="Print", daemon=True).start()


# ---------------------------------------------------------------------------
# Payment trigger HTTP server
# ---------------------------------------------------------------------------

TRIGGER_PORT = 8080  # POST /trigger → injects a start_note MIDI event

def _make_unlock_handler(credits_ref: list, credits_lock: "threading.Lock",
                         midi_queue: "queue.Queue", credits_per_unlock: int = 1):
    """
    POST /unlock  → grants credits_per_unlock photo credits (called by payment server on success)
    GET  /credits → returns current credit count (for debugging)
    POST /trigger → injects a virtual MIDI start_note (to trigger capture headlessly)
    """
    class UnlockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/unlock":
                with credits_lock:
                    credits_ref[0] += credits_per_unlock
                    current = credits_ref[0]
                logging.info("Payment received — credits now: %d", current)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "credits": current}).encode())
            elif self.path == "/trigger":
                logging.info("Headless trigger received via HTTP")
                # note=60 matches Leonardo middle C
                midi_queue.put(MidiEvent(type="note_on", note=60, velocity=100))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/credits":
                with credits_lock:
                    current = credits_ref[0]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"credits": current}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):  # silence default access log
            pass

    return UnlockHandler


def start_unlock_server(credits_ref: list, credits_lock: "threading.Lock",
                        midi_queue: "queue.Queue", credits_per_unlock: int = 1) -> None:
    handler = _make_unlock_handler(credits_ref, credits_lock, midi_queue, credits_per_unlock)
    server  = HTTPServer(("0.0.0.0", TRIGGER_PORT), handler)
    t = threading.Thread(target=server.serve_forever, name="UnlockHTTP", daemon=True)
    t.start()
    logging.info("Payment unlock server listening on http://127.0.0.1:%d/unlock", TRIGGER_PORT)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class PhotoboothApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg     = cfg
        self.running = True

        self.filter_name       = (cfg.default_filter if cfg.default_filter in cfg.filter_cycle
                                  else cfg.filter_cycle[0])
        self.brightness        = float(cfg.brightness)
        self.countdown_seconds = int(cfg.countdown_seconds)

        self.last_photo_path: Optional[Path] = None
        self.last_photo_shown_until: float   = 0.0

        self.status_message = ""
        self.status_until   = 0.0

        self.delete_confirm_mode  = False
        self.delete_confirm_until = 0.0

        self.midi_queue: "queue.Queue[MidiEvent]" = queue.Queue()
        self.midi = MidiController(cfg.midi_port_name, self.midi_queue)
        self.midi.start()

        self._payment_credits: list = [0]       # mutable for cross-thread access
        self._credits_lock = threading.Lock()
        start_unlock_server(self._payment_credits, self._credits_lock, self.midi_queue,
                            credits_per_unlock=int(cfg.credits_per_unlock))

        self.camera  = CameraController(cfg.camera)
        self.printer = PrintManager(
            copies=cfg.copies,
            image_reduce_factor=cfg.image_reduce_factor,
            qr_url=cfg.qr_url,
            print_brightness=cfg.print_brightness,
            print_contrast=cfg.print_contrast,
        )
        self.web_uploader = WebUploader(
            upload_url=cfg.upload_url,
            upload_token=cfg.upload_token,
            queue_dir=Path("./prints_queue_web"),
        )
        if self.web_uploader.enabled:
            logging.info("Web uploader enabled → %s", cfg.upload_url)
        else:
            logging.info("Web uploader disabled (no url/token); QR will use printer.qr_url")
        self._last_camera_ok = time.time()

        # Audio
        self.snd_countdown: Optional[pygame.mixer.Sound] = None
        self.snd_shutter:   Optional[pygame.mixer.Sound] = None
        self._ch_countdown: Optional[pygame.mixer.Channel] = None
        self._ch_shutter:   Optional[pygame.mixer.Channel] = None

        try:
            pygame.mixer.pre_init(48000, -16, 2, 512)
        except Exception:
            pass

        pygame.init()
        pygame.font.init()

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            self._ch_countdown = pygame.mixer.Channel(0)
            self._ch_shutter   = pygame.mixer.Channel(1)
            self.snd_countdown = safe_load_sound(cfg.countdown_wav)
            self.snd_shutter   = safe_load_sound(cfg.shutter_wav)
            if self.snd_countdown: self.snd_countdown.set_volume(0.35)
            if self.snd_shutter:   self.snd_shutter.set_volume(0.50)
            if cfg.idle_music:
                music_path = Path(cfg.idle_music)
                if music_path.exists():
                    pygame.mixer.music.load(str(music_path))
                    pygame.mixer.music.play(-1)  # -1 = loop forever
                    logging.info("Idle music started: %s", music_path)
                else:
                    logging.warning("idle_music not found: %s", music_path)
        except Exception:
            logging.exception("Audio init failed (non-fatal).")

        self.screen = init_pygame_display(cfg.fullscreen)
        pygame.display.set_caption(APP_NAME)
        self.w, self.h = self.screen.get_size()
        pygame.mouse.set_visible(False)

        self.font_big   = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.14), bold=True)
        self.font_med   = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.052), bold=True)
        self.font_small = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.034))
        self.font_hud   = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.030), bold=True)

        self._scanlines = make_scanlines((self.w, self.h), alpha=Y2K["scanline_alpha"])
        self._vignette  = make_vignette((self.w, self.h))
        self._t0        = time.time()

        self._thumb_cache_path: Optional[Path]           = None
        self._thumb_cache_surf: Optional[pygame.Surface] = None

        self._pay_overlay_surf: Optional[pygame.Surface] = None
        self._credits_prev: int = 0
        self._payment_accepted_until: float = 0.0
        self._in_session: bool = False
        if cfg.pay_overlay_image:
            try:
                p = Path(cfg.pay_overlay_image).expanduser()
                if not p.is_absolute():
                    p = (Path(__file__).resolve().parent / p).resolve()
                if p.exists():
                    logo = pygame.image.load(str(p)).convert_alpha()
                    max_w = int(self.w * 0.55)
                    max_h = int(self.h * 0.55)
                    lw, lh = logo.get_size()
                    ratio = min(max_w / lw, max_h / lh)
                    self._pay_overlay_surf = pygame.transform.smoothscale(
                        logo, (int(lw * ratio), int(lh * ratio))
                    )
                    logging.info("Pay overlay loaded: %s", p)
                else:
                    logging.warning("pay_overlay_image not found: %s", p)
            except Exception:
                logging.exception("Failed to load pay_overlay_image")

        self.clock = pygame.time.Clock()

    # ---- lifecycle ----

    def shutdown(self) -> None:
        self.running = False
        try: self.midi.stop()
        except Exception: pass
        try: self.camera.close()
        except Exception: pass
        try: self.web_uploader.stop()
        except Exception: pass
        pygame.quit()

    # ---- payment credit helpers ----

    def _consume_credit(self) -> bool:
        """Returns True and decrements if a credit is available, False otherwise."""
        with self._credits_lock:
            if self._payment_credits[0] > 0:
                self._payment_credits[0] -= 1
                return True
        return False

    def _has_credit(self) -> bool:
        with self._credits_lock:
            return self._payment_credits[0] > 0

    # ---- status / helpers ----

    def set_status(self, msg: str, seconds: float = 2.0) -> None:
        self.status_message = msg
        self.status_until   = time.time() + seconds
        logging.info("STATUS: %s", msg)

    def _play(self, ch: Optional[pygame.mixer.Channel],
              snd: Optional[pygame.mixer.Sound]) -> None:
        if snd is None:
            return
        try:
            if ch:
                ch.stop()
                ch.play(snd)
            else:
                snd.play()
        except Exception:
            pass

    def _music_pause(self) -> None:
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
        except Exception:
            pass

    def _music_resume(self) -> None:
        try:
            pygame.mixer.music.unpause()
        except Exception:
            pass

    # ---- MIDI actions ----

    def cycle_filter(self) -> None:
        cycle = list(self.cfg.filter_cycle)
        idx = cycle.index(self.filter_name) if self.filter_name in cycle else 0
        self.filter_name = cycle[(idx + 1) % len(cycle)]
        self.set_status(f"Filter: {self.filter_name}", 1.3)

    def handle_midi_event(self, ev: MidiEvent) -> None:
        mm = self.cfg.midi_mapping
        if ev.type == "note_on" and ev.note is not None:
            if ev.note == mm.start_note and not self.delete_confirm_mode:
                if self._consume_credit():
                    self._start_photo_sequence(ev)
            elif ev.note == mm.filter_note and not self.delete_confirm_mode:
                self.cycle_filter()
            elif ev.note == mm.print_note and not self.delete_confirm_mode:
                self._print_last()
            elif ev.note == mm.delete_note:
                self._delete_last()
        elif ev.type == "control_change" and ev.control is not None and ev.value is not None:
            if ev.control == mm.brightness_cc:
                self.brightness = 0.5 + clamp(ev.value / 127.0, 0.0, 1.0) * 1.3
                self.set_status(f"Brightness: {self.brightness:.2f}", 1.0)
            elif ev.control == mm.countdown_cc:
                self.countdown_seconds = int(round(1 + clamp(ev.value / 127.0, 0.0, 1.0) * 9))
                self.set_status(f"Countdown: {self.countdown_seconds}s", 1.0)

    def _print_last(self) -> None:
        if not self.last_photo_path or not self.last_photo_path.exists():
            self.set_status("No photo to print", 2.0)
            return
        if self.printer.is_printing():
            self.set_status("Already printing…", 1.5)
            return
        self.camera.close()  # free USB bandwidth during printing
        self.set_status("Printing…", 120.0)
        self.printer.print_async(
            self.last_photo_path,
            on_done=lambda ok, msg: self.set_status(msg, 3.0),
        )

    def _delete_last(self) -> None:
        if not self.last_photo_path or not self.last_photo_path.exists():
            self.set_status("No photo to delete", 2.0)
            return
        now = time.time()
        if not self.delete_confirm_mode:
            self.delete_confirm_mode  = True
            self.delete_confirm_until = now + 5.0
            self.set_status("Press DELETE again to confirm", 5.0)
            return
        if now <= self.delete_confirm_until:
            try:
                p = self.last_photo_path
                p.with_suffix(".json").unlink(missing_ok=True)
                p.unlink(missing_ok=True)
                self.last_photo_path   = None
                self._thumb_cache_path = None
                self._thumb_cache_surf = None
                self.set_status("Deleted", 2.0)
            except Exception as e:
                logging.exception("Delete failed: %s", e)
                self.set_status("Delete failed", 2.5)
        else:
            self.set_status("Delete timed out", 2.0)
        self.delete_confirm_mode = False

    # ---- photo sequence ----

    def _start_photo_sequence(self, trigger: MidiEvent) -> None:
        if not self.camera.available():
            self.set_status("Camera unavailable, retrying…", 2.0)
            self.camera.retry()
            return

        self._music_pause()
        self._in_session = True

        n = max(1, int(self.cfg.photos_per_session))
        session_dt = now_dt()
        day_dir    = PHOTOS_DIR / today_folder_name(session_dt)
        day_dir.mkdir(parents=True, exist_ok=True)
        session_id   = timestamp_name(session_dt)
        ticket_code  = generate_ticket_code()
        photo_paths: list[Path] = []

        try:
            for i in range(1, n + 1):
                self.set_status(f"Photo {i}/{n}", float(self.countdown_seconds))

                for t in range(self.countdown_seconds, 0, -1):
                    self._play(self._ch_countdown, self.snd_countdown)
                    self._draw_frame(countdown=t)
                    pygame.display.flip()
                    self.clock.tick(60)
                    time.sleep(1.0)

                self._play(self._ch_shutter, self.snd_shutter)

                # Keep camera running across shots; only stop after the last
                # so its USB isochronous transfers don't compete with the printer.
                if i < n:
                    img = self.camera.capture_still()
                else:
                    img = self.camera.capture_still_and_stop()

                if img is None:
                    self.set_status(f"Capture {i}/{n} failed", 2.0)
                    logging.warning("Capture %d/%d returned None", i, n)
                    continue

                shot_dt    = now_dt()
                fname      = f"{session_id}_{ticket_code}_{i:02d}of{n:02d}_{self.filter_name}.jpg"
                photo_path = day_dir / fname
                try:
                    processed = apply_filter(img, self.filter_name, self.brightness)
                    processed.save(photo_path, format="JPEG", quality=95)
                    safe_write_json(photo_path.with_suffix(".json"), {
                        "timestamp_iso":  shot_dt.isoformat(),
                        "photo_path":     str(photo_path),
                        "session_id":     session_id,
                        "ticket_code":    ticket_code,
                        "session_index":  i,
                        "session_total":  n,
                        "filter":         self.filter_name,
                        "brightness":     self.brightness,
                        "countdown":      self.countdown_seconds,
                        "camera_backend": self.camera.backend,
                        "midi_trigger":   trigger.raw or {},
                    })
                    photo_paths.append(photo_path)
                    self.last_photo_path        = photo_path
                    self.last_photo_shown_until = time.time() + float(self.cfg.between_shots_seconds)
                    self._thumb_cache_path      = None
                    self._thumb_cache_surf      = None
                    logging.info("Saved %d/%d: %s", i, n, photo_path)
                except Exception as e:
                    logging.exception("Save %d/%d failed: %s", i, n, e)
                    self.set_status(f"Save {i}/{n} failed", 2.0)

                # Short pause so the user sees their shot before the next countdown.
                # Skip after the last since we're about to print.
                if i < n:
                    pause_until = time.time() + float(self.cfg.between_shots_seconds)
                    while time.time() < pause_until:
                        self._draw_frame()
                        self._draw_ready_overlay()
                        pygame.display.flip()
                        self.clock.tick(30)

            if not photo_paths:
                self.set_status("Session failed", 3.0)
                self.camera.retry()
                return

            # Compute per-session QR URL from the ticket code.
            ticket_url = ""
            if self.web_uploader.enabled and self.cfg.ticket_url_template:
                try:
                    ticket_url = self.cfg.ticket_url_template.format(code=ticket_code)
                except Exception:
                    ticket_url = ""
                # Kick off upload immediately; runs in parallel with print.
                self.web_uploader.queue(ticket_code, photo_paths)

            # Camera is already stopped by capture_still_and_stop() for the last shot.
            # Small USB cool-down before the printer bulk transfers begin.
            time.sleep(1.0)

            self.set_status("Printing…", 120.0)
            self.printer.print_async(
                photo_paths,
                on_done=lambda ok, msg: self.set_status(msg, 3.0),
                qr_url=ticket_url or None,
            )

        finally:
            self._in_session = False
            self._music_resume()

    # ---- drawing ----

    def _draw_ready_overlay(self) -> None:
        """Pulsing 'PRÊT' shown between captures within a session."""
        text = self.cfg.ready_text
        if not text:
            return
        pulse = 0.5 + 0.5 * math.sin((time.time() - self._t0) * 4.0)
        draw_glow_text(
            self.screen, text, self.font_big,
            (self.w // 2, int(self.h * 0.46)),
            Y2K["text"], Y2K["accent_lime"],
            center=True, glow_px=10, glow_layers=7,
            alpha_max=int(220 * pulse),
        )

    def _draw_payment_accepted(self) -> None:
        """Brief confirmation after /unlock, before handing off to the live preview."""
        # Lighter backdrop so the preview shows through and feels alive.
        dim = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 110))
        self.screen.blit(dim, (0, 0))

        # Fade out over the last third of the window for a smooth handoff.
        remaining = max(0.0, self._payment_accepted_until - time.time())
        total = max(0.1, float(self.cfg.payment_accepted_seconds))
        fade = min(1.0, remaining / (total / 3.0))

        text = self.cfg.payment_accepted_text
        if text:
            draw_glow_text(
                self.screen, text, self.font_big,
                (self.w // 2, int(self.h * 0.44)),
                Y2K["text"], Y2K["accent_lime"],
                center=True, glow_px=10, glow_layers=7,
                alpha_max=int(200 * fade),
            )
        # Credits available, as a sub-line
        sub = f"{self._credits_prev} PHOTOS"
        draw_glow_text(
            self.screen, sub, self.font_med,
            (self.w // 2, int(self.h * 0.58)),
            Y2K["text"], Y2K["accent_cyan"],
            center=True, glow_px=5, glow_layers=4,
            alpha_max=int(170 * fade),
        )

    def _draw_pay_overlay(self) -> None:
        """Pay-first screen: dim backdrop, blinking logo + price, solid footer."""
        # Dim backdrop
        dim = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        # Blink — sin wave from 0.45 to 1.0
        pulse = 0.725 + 0.275 * math.sin((time.time() - self._t0) * 3.2)

        # Logo
        logo_bottom_y = int(self.h * 0.36)
        if self._pay_overlay_surf is not None:
            logo = self._pay_overlay_surf.copy()
            logo.set_alpha(int(255 * pulse))
            rect = logo.get_rect(center=(self.w // 2, int(self.h * 0.38)))
            self.screen.blit(logo, rect)
            logo_bottom_y = rect.bottom

        # Price text, blinking in sync with logo
        price = self.cfg.pay_overlay_price_text
        if price:
            draw_glow_text(
                self.screen, price, self.font_big,
                (self.w // 2, logo_bottom_y + int(self.h * 0.10)),
                Y2K["text"], Y2K["accent_magenta"],
                center=True, glow_px=8, glow_layers=6,
                alpha_max=int(180 * pulse),
            )

        # Footer (solid, not blinking)
        footer = self.cfg.pay_overlay_footer_text
        if footer:
            draw_glow_text(
                self.screen, footer, self.font_med,
                (self.w // 2, self.h - int(self.h * 0.08)),
                Y2K["text"], Y2K["accent_cyan"],
                center=True, glow_px=4, glow_layers=4,
            )

    def _rotate(self, rgb: np.ndarray) -> np.ndarray:
        deg = self.cfg.rotate_preview_degrees % 360
        k = {90: 3, 180: 2, 270: 1}.get(deg)
        return np.rot90(rgb, k=k) if k else rgb

    def _brightness_array(self, rgb: np.ndarray) -> np.ndarray:
        if abs(self.brightness - 1.0) < 0.01:
            return rgb
        return np.clip(rgb.astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)

    def _draw_frame(self, countdown: Optional[int] = None) -> None:
        self.screen.fill((0, 0, 0))

        # Show last photo for a few seconds after capture
        if (self.last_photo_path and self.last_photo_path.exists()
                and time.time() < self.last_photo_shown_until):
            try:
                img  = Image.open(self.last_photo_path).convert("RGB")
                img  = ImageOps.contain(img, (self.w, self.h))
                arr  = np.array(img)
                surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                self.screen.blit(surf, surf.get_rect(center=(self.w // 2, self.h // 2)))
                self.screen.blit(self._scanlines, (0, 0))
                self.screen.blit(self._vignette,  (0, 0))
                return
            except Exception:
                pass

        # Live preview
        frame = self.camera.get_preview_frame() if self.camera.available() else None
        if frame is None:
            if time.time() - self._last_camera_ok > 2.0 and not self.printer.is_printing():
                self.camera.retry()
                self._last_camera_ok = time.time()
            if not self.cfg.minimal_ui:
                draw_glow_text(self.screen, "CAMERA ERROR", self.font_big,
                               (self.w // 2, int(self.h * 0.38)),
                               (255, 200, 200), (255, 80, 80), center=True)
        else:
            self._last_camera_ok = time.time()
            frame = self._rotate(frame)
            frame = self._brightness_array(frame)
            img   = ImageOps.contain(Image.fromarray(frame, "RGB"), (self.w, self.h))
            arr   = np.array(img)
            surf  = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            self.screen.blit(surf, surf.get_rect(center=(self.w // 2, self.h // 2)))

        # Countdown overlay
        if countdown is not None:
            t     = time.time()
            pulse = 0.5 + 0.5 * math.sin((t - self._t0) * 2.8)
            bw    = max(180, int(self.w * 0.22))
            bh    = max(180, int(self.h * 0.30))
            box   = pygame.Rect(self.w // 2 - bw // 2,
                                int(self.h * 0.42) - bh // 2, bw, bh)
            draw_bevel_panel(self.screen, box, (8, 10, 16, 120),
                             (60, 255, 220, int(120 + 60 * pulse)),
                             14, 2, 2)
            draw_glow_text(self.screen, str(countdown), self.font_big,
                           (self.w // 2, int(self.h * 0.42)),
                           Y2K["text"], Y2K["accent_cyan"],
                           center=True, glow_px=10, glow_layers=7)

        # Detect 0→positive transition so we can show a brief "accepted" message.
        with self._credits_lock:
            credits_now = self._payment_credits[0]
        if credits_now > self._credits_prev:
            self._payment_accepted_until = time.time() + float(self.cfg.payment_accepted_seconds)
        self._credits_prev = credits_now

        # Overlays suppressed during an active session (countdown, capture, between-shot display).
        if countdown is None and not self._in_session:
            if time.time() < self._payment_accepted_until:
                self._draw_payment_accepted()
            elif credits_now == 0:
                self._draw_pay_overlay()

        # Status message (shown even in minimal_ui)
        if self.status_message and time.time() < self.status_until:
            draw_glow_text(self.screen, self.status_message, self.font_hud,
                           (self.w // 2, self.h - int(self.h * 0.06)),
                           Y2K["text"], Y2K["accent_cyan"],
                           center=True, glow_px=4, glow_layers=4)

        self.screen.blit(self._scanlines, (0, 0))
        self.screen.blit(self._vignette,  (0, 0))

    # ---- main loop ----

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        if self._consume_credit():
                            self._start_photo_sequence(
                                MidiEvent(type="manual", raw={"type": "manual"}))
                    elif event.key == pygame.K_f:
                        self.cycle_filter()
                    elif event.key == pygame.K_p:
                        self._print_last()

            try:
                while True:
                    self.handle_midi_event(self.midi_queue.get_nowait())
            except queue.Empty:
                pass

            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(30)

        self.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log_path = Path(__file__).parent / "logs" / "photobooth.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    ensure_dirs()

    if "--list-midi" in sys.argv:
        ports = MidiController.list_ports()
        print("MIDI input ports:")
        for p in ports: print(f"  {p}")
        if not ports: print("  (none)")
        return

    cfg = load_config()
    logging.info("Starting %s | midi=%s | fullscreen=%s",
                 APP_NAME, cfg.midi_port_name or "auto", cfg.fullscreen)

    app = PhotoboothApp(cfg)
    logging.info("Display driver: %s | camera: %s",
                 pygame.display.get_driver(), app.camera.backend)
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    main()
