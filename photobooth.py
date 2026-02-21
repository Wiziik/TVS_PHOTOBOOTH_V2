#!/usr/bin/env python3
"""
PiPhotobooth — Raspberry Pi photobooth with:
- Fullscreen preview (pygame)
- Countdown overlay
- High-res still capture (Picamera2/libcamera preferred)
- Filters (none, bw, retro)
- Save with timestamp + JSON sidecar metadata
- Print via CUPS (lp) OR enqueue to ./prints_queue
- MIDI controls via mido + python-rtmidi (robust reconnect)

Patched:
- photobooth.ini support (configparser) + legacy config.json fallback
- minimal_ui preview-only mode (no HUD text/panels)
- countdown_wav tick + optional shutter_wav
- MIDI auto-pick avoids "Midi Through" when other ports exist
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import logging
import math
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

# Hide pygame banner (must be set before importing pygame)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import mido
import pygame


APP_NAME = "PiPhotobooth"

DEFAULT_INI_PATH = Path("./photobooth.ini")
DEFAULT_JSON_PATH = Path("./config.json")
DEFAULT_CONFIG_PATH = DEFAULT_INI_PATH  # prefer INI

PHOTOS_DIR = Path("./photos")
PRINT_QUEUE_DIR = Path("./prints_queue")


# ----------------------------
# Y2K UI Theme (pygame-only)
# ----------------------------

Y2K = {
    "panel_fill": (10, 12, 18, 150),
    "panel_fill_2": (18, 20, 30, 165),
    "panel_stroke": (140, 190, 255, 180),
    "panel_stroke_hot": (255, 70, 200, 200),
    "panel_stroke_ok": (60, 255, 200, 200),
    "panel_stroke_warn": (255, 170, 60, 210),
    "panel_stroke_err": (255, 80, 80, 220),
    "text": (235, 245, 255),
    "text_dim": (190, 205, 220),
    "accent_cyan": (60, 255, 220),
    "accent_magenta": (255, 70, 200),
    "accent_lime": (160, 255, 90),
    "scanline_alpha": 18,
}


def draw_bevel_panel(
    target: pygame.Surface,
    rect: pygame.Rect,
    fill_rgba: tuple[int, int, int, int],
    stroke_rgba: tuple[int, int, int, int],
    radius: int,
    stroke: int,
    bevel: int = 2,
) -> None:
    """Lightweight glossy / beveled panel using SRCALPHA surfaces."""
    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)

    pygame.draw.rect(panel, fill_rgba, panel.get_rect(), border_radius=radius)

    # glossy top band
    top_h = max(2, rect.h // 3)
    gloss = pygame.Surface((rect.w, top_h), pygame.SRCALPHA)
    pygame.draw.rect(gloss, (255, 255, 255, 24), gloss.get_rect(), border_radius=radius)
    panel.blit(gloss, (0, 0))

    # bevel highlight/shadow
    pygame.draw.rect(panel, (255, 255, 255, 32), panel.get_rect(), width=bevel, border_radius=radius)
    inset = panel.get_rect().inflate(-bevel * 2, -bevel * 2)
    pygame.draw.rect(panel, (0, 0, 0, 40), inset, width=bevel, border_radius=max(0, radius - bevel))

    # stroke
    pygame.draw.rect(panel, stroke_rgba, panel.get_rect(), width=stroke, border_radius=radius)

    target.blit(panel, rect.topleft)


def draw_glow_text(
    target: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    glow_color: tuple[int, int, int],
    *,
    center: bool = False,
    glow_px: int = 5,
    glow_layers: int = 6,
    alpha_max: int = 170,
) -> pygame.Rect:
    """Cheap glow: render text multiple times with small offsets + alpha, then final crisp text."""
    base = font.render(text, True, color)
    rect = base.get_rect(center=pos) if center else base.get_rect(topleft=pos)

    for i in range(glow_layers, 0, -1):
        a = int(alpha_max * (i / glow_layers) * (i / glow_layers))
        glow = font.render(text, True, glow_color).convert_alpha()
        glow.set_alpha(a)
        off = int((i / glow_layers) * glow_px)
        for dx, dy in (
            (-off, 0),
            (off, 0),
            (0, -off),
            (0, off),
            (-off, -off),
            (-off, off),
            (off, -off),
            (off, off),
        ):
            target.blit(glow, rect.move(dx, dy))

    target.blit(base, rect)
    return rect


def make_scanlines(size: tuple[int, int], alpha: int = 18, spacing: int = 3) -> pygame.Surface:
    """Prebaked scanline overlay surface (subtle)."""
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    line = pygame.Surface((w, 1), pygame.SRCALPHA)
    line.fill((0, 0, 0, alpha))
    for y in range(0, h, spacing):
        surf.blit(line, (0, y))
    return surf


def make_vignette(size: tuple[int, int], strength: int = 55) -> pygame.Surface:
    """Very cheap vignette: 4 translucent rects, no per-pixel ops."""
    w, h = size
    v = pygame.Surface((w, h), pygame.SRCALPHA)
    pad = max(12, min(w, h) // 18)
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, 0, w, pad))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, h - pad, w, pad))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(0, 0, pad, h))
    pygame.draw.rect(v, (0, 0, 0, strength), pygame.Rect(w - pad, 0, pad, h))
    return v


def auto_set_sdl_videodriver() -> None:
    """
    Auto-pick a safe SDL driver when running without DISPLAY/WAYLAND.
    Note: On Debian 13 you often need to launch from a real desktop session.
    """
    if os.environ.get("SDL_VIDEODRIVER"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    if Path("/dev/fb0").exists():
        os.environ["SDL_VIDEODRIVER"] = "fbcon"
        os.environ.setdefault("SDL_FBDEV", "/dev/fb0")
    elif Path("/dev/dri").exists():
        os.environ["SDL_VIDEODRIVER"] = "kmsdrm"


def init_pygame_display(fullscreen: bool) -> pygame.Surface:
    """
    Initialize pygame display with sane fallbacks.
    Tries current SDL_VIDEODRIVER first, then common drivers.
    """
    flags = pygame.FULLSCREEN if fullscreen else 0

    # Try the currently-selected driver first
    try:
        return pygame.display.set_mode((0, 0), flags)
    except pygame.error as e:
        logging.warning(
            "pygame set_mode failed for driver=%s: %s",
            os.environ.get("SDL_VIDEODRIVER"),
            e,
        )

    candidates: list[tuple[str, Dict[str, str]]] = [
        ("kmsdrm", {"SDL_VIDEO_KMSDRM_DEVICE": "/dev/dri/card0"}),
        ("fbcon", {"SDL_FBDEV": "/dev/fb0"}),
        ("wayland", {}),
        ("x11", {}),
    ]

    for drv, extra_env in candidates:
        try:
            os.environ["SDL_VIDEODRIVER"] = drv
            for k, v in extra_env.items():
                os.environ.setdefault(k, v)
            pygame.display.quit()
            pygame.display.init()
            return pygame.display.set_mode((0, 0), flags)
        except pygame.error as e:
            logging.warning("pygame set_mode failed for fallback driver=%s: %s", drv, e)

    raise pygame.error("No usable SDL video driver found (kmsdrm/fbcon/wayland/x11 all failed).")


# ----------------------------
# Utilities
# ----------------------------

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


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def safe_load_sound(path: str) -> Optional["pygame.mixer.Sound"]:
    """Load a WAV safely (returns None if missing/unloadable)."""
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


# ----------------------------
# Config
# ----------------------------

@dataclass
class MidiMapping:
    start_note: int = 36
    filter_note: int = 38
    print_note: int = 40
    delete_note: int = 41
    brightness_cc: int = 1
    countdown_cc: int = 2


@dataclass
class CameraConfig:
    # Picamera2
    picamera2_preview_size: Tuple[int, int] = (1280, 720)
    picamera2_still_size: Tuple[int, int] = (3280, 2464)
    picamera2_swap_rb: bool = False
    # OpenCV fallback
    opencv_device_index: int = 0
    opencv_width: int = 1280
    opencv_height: int = 720


@dataclass
class AppConfig:
    midi_port_name: str = ""  # substring match; empty => auto-pick (avoids Midi Through)
    midi_mapping: MidiMapping = dataclasses.field(default_factory=MidiMapping)

    fullscreen: bool = True
    rotate_preview_degrees: int = 0  # 0/90/180/270

    countdown_seconds: int = 3
    last_photo_show_seconds: int = 3
    brightness: float = 1.0

    filter_cycle: Tuple[str, ...] = ("none", "bw", "retro")
    default_filter: str = "none"

    prefer_cups: bool = True
    cups_printer_name: str = ""
    copies: int = 1

    # UI
    minimal_ui: bool = True  # preview only, no HUD text/panels

    # Sounds
    countdown_wav: str = "./sounds/countdown.wav"  # tick each second
    shutter_wav: str = ""  # optional

    camera: CameraConfig = dataclasses.field(default_factory=CameraConfig)


def default_config_dict() -> Dict[str, Any]:
    return dataclasses.asdict(AppConfig())


def _ini_bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _ini_int(v: str | None, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _ini_float(v: str | None, default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def default_ini_text() -> str:
    return """\
[app]
fullscreen = true
minimal_ui = true
rotate_preview_degrees = 0

countdown_seconds = 3
last_photo_show_seconds = 3
brightness = 1.0

countdown_wav = ./sounds/countdown.wav
shutter_wav =

[filters]
cycle = none,bw,retro
default = none

[midi]
port_name = Akai MPK61 Port 1

start_note = 36
filter_note = 38
print_note = 40
delete_note = 41

brightness_cc = 1
countdown_cc = 2

[printer]
prefer_cups = true
printer_name =
copies = 1

[camera]
picamera2_preview_width = 1280
picamera2_preview_height = 720

picamera2_still_width = 3280
picamera2_still_height = 2464

picamera2_swap_rb = false

opencv_device_index = 0
opencv_width = 1280
opencv_height = 720
"""


def load_or_create_ini_config(path: Path) -> AppConfig:
    if not path.exists():
        path.write_text(default_ini_text(), encoding="utf-8")
        logging.info("Created default INI config at %s", path)

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding="utf-8")

    app = cp["app"] if cp.has_section("app") else {}
    filt = cp["filters"] if cp.has_section("filters") else {}
    midi = cp["midi"] if cp.has_section("midi") else {}
    prn = cp["printer"] if cp.has_section("printer") else {}
    cam = cp["camera"] if cp.has_section("camera") else {}

    cycle_raw = str(filt.get("cycle", "none,bw,retro"))
    filter_cycle = tuple([x.strip() for x in cycle_raw.split(",") if x.strip()] or ["none"])
    default_filter = str(filt.get("default", filter_cycle[0])).strip() or filter_cycle[0]

    mm = MidiMapping(
        start_note=_ini_int(midi.get("start_note"), 36),
        filter_note=_ini_int(midi.get("filter_note"), 38),
        print_note=_ini_int(midi.get("print_note"), 40),
        delete_note=_ini_int(midi.get("delete_note"), 41),
        brightness_cc=_ini_int(midi.get("brightness_cc"), 1),
        countdown_cc=_ini_int(midi.get("countdown_cc"), 2),
    )

    preview_w = _ini_int(cam.get("picamera2_preview_width"), 1280)
    preview_h = _ini_int(cam.get("picamera2_preview_height"), 720)
    still_w = _ini_int(cam.get("picamera2_still_width"), 3280)
    still_h = _ini_int(cam.get("picamera2_still_height"), 2464)

    cam_cfg = CameraConfig(
        picamera2_preview_size=(preview_w, preview_h),
        picamera2_still_size=(still_w, still_h),
        picamera2_swap_rb=_ini_bool(cam.get("picamera2_swap_rb"), False),
        opencv_device_index=_ini_int(cam.get("opencv_device_index"), 0),
        opencv_width=_ini_int(cam.get("opencv_width"), 1280),
        opencv_height=_ini_int(cam.get("opencv_height"), 720),
    )

    return AppConfig(
        midi_port_name=str(midi.get("port_name", "")).strip(),
        midi_mapping=mm,

        fullscreen=_ini_bool(app.get("fullscreen"), True),
        rotate_preview_degrees=_ini_int(app.get("rotate_preview_degrees"), 0),

        countdown_seconds=_ini_int(app.get("countdown_seconds"), 3),
        last_photo_show_seconds=_ini_int(app.get("last_photo_show_seconds"), 3),
        brightness=_ini_float(app.get("brightness"), 1.0),

        filter_cycle=filter_cycle,
        default_filter=default_filter,

        prefer_cups=_ini_bool(prn.get("prefer_cups"), True),
        cups_printer_name=str(prn.get("printer_name", "")).strip(),
        copies=_ini_int(prn.get("copies"), 1),

        minimal_ui=_ini_bool(app.get("minimal_ui"), True),

        countdown_wav=str(app.get("countdown_wav", "./sounds/countdown.wav")).strip() or "./sounds/countdown.wav",
        shutter_wav=str(app.get("shutter_wav", "")).strip(),

        camera=cam_cfg,
    )


def load_or_create_config(path: Path) -> AppConfig:
    """
    Prefer INI if present. Fall back to JSON if INI missing.
    """
    if path.suffix.lower() == ".ini" or DEFAULT_INI_PATH.exists():
        return load_or_create_ini_config(DEFAULT_INI_PATH)

    # Legacy JSON
    if not path.exists():
        safe_write_json(path, default_config_dict())
        logging.info("Created default JSON config at %s", path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    mm = MidiMapping(**raw.get("midi_mapping", {}))
    cam = CameraConfig(**raw.get("camera", {}))

    return AppConfig(
        midi_port_name=str(raw.get("midi_port_name", "")),
        midi_mapping=mm,
        fullscreen=bool(raw.get("fullscreen", True)),
        rotate_preview_degrees=int(raw.get("rotate_preview_degrees", 0)),
        countdown_seconds=int(raw.get("countdown_seconds", 3)),
        last_photo_show_seconds=int(raw.get("last_photo_show_seconds", 3)),
        brightness=float(raw.get("brightness", 1.0)),
        filter_cycle=tuple(raw.get("filter_cycle", ("none", "bw", "retro"))),
        default_filter=str(raw.get("default_filter", "none")),
        prefer_cups=bool(raw.get("prefer_cups", True)),
        cups_printer_name=str(raw.get("cups_printer_name", "")),
        copies=int(raw.get("copies", 1)),
        minimal_ui=bool(raw.get("minimal_ui", True)),
        countdown_wav=str(raw.get("countdown_wav", "./sounds/countdown.wav")),
        shutter_wav=str(raw.get("shutter_wav", "")),
        camera=cam,
    )


# ----------------------------
# MIDI
# ----------------------------

@dataclass
class MidiEvent:
    type: str
    note: Optional[int] = None
    velocity: Optional[int] = None
    control: Optional[int] = None
    value: Optional[int] = None
    port: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class MidiController:
    """Connects to a MIDI input (substring match), retries on disconnect, pushes events into a queue."""

    def __init__(self, port_name: str, out_queue: "queue.Queue[MidiEvent]", poll_interval: float = 2.0) -> None:
        self.port_name = port_name
        self.out_queue = out_queue
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="MidiController", daemon=True)
        self._connected_port: Optional[str] = None

    @staticmethod
    def list_ports() -> list[str]:
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def is_connected(self) -> bool:
        return self._connected_port is not None

    def connected_port(self) -> Optional[str]:
        return self._connected_port

    def _pick_port(self, names: list[str]) -> Optional[str]:
        if not names:
            return None

        # If user specified a port substring, honor it.
        if self.port_name:
            needle = self.port_name.lower()
            for n in names:
                if needle in n.lower():
                    return n
            return None

        # Otherwise auto-pick: avoid "Midi Through" if there are other ports.
        non_through = [n for n in names if "through" not in n.lower()]
        if non_through:
            return non_through[0]
        return names[0]

    def _normalize(self, msg: mido.Message, port: str) -> Optional[MidiEvent]:
        d = msg.dict()
        t = d.get("type")
        if t == "note_on":
            if int(d.get("velocity", 0)) <= 0:
                return None
            return MidiEvent(type="note_on", note=int(d["note"]), velocity=int(d["velocity"]), port=port, raw=d)
        if t == "control_change":
            return MidiEvent(type="control_change", control=int(d["control"]), value=int(d["value"]), port=port, raw=d)
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                names = self.list_ports()
                chosen = self._pick_port(names)
                if not chosen:
                    self._connected_port = None
                    time.sleep(self.poll_interval)
                    continue

                logging.info("Connecting MIDI input: %s", chosen)
                with mido.open_input(chosen) as port:
                    self._connected_port = chosen
                    for msg in port:
                        if self._stop.is_set():
                            break
                        ev = self._normalize(msg, chosen)
                        if ev:
                            self.out_queue.put(ev)

                logging.warning("MIDI port closed/disconnected: %s", chosen)
                self._connected_port = None

            except Exception as e:
                logging.exception("MIDI error: %s", e)
                self._connected_port = None
                time.sleep(self.poll_interval)


# ----------------------------
# Camera
# ----------------------------

class CameraController:
    """
    - get_preview_frame(): RGB numpy array
    - capture_still(): high-res PIL.Image (RGB)
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.backend = "none"

        self._picam2 = None
        self._preview_config = None
        self._still_config = None

        self._opencv_cap = None

        self._init_camera()

    def _init_camera(self) -> None:
        # Prefer Picamera2
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam2 = Picamera2()
            pw, ph = self.cfg.picamera2_preview_size
            sw, sh = self.cfg.picamera2_still_size

            self._preview_config = self._picam2.create_preview_configuration(
                main={"size": (pw, ph), "format": "RGB888"},
                controls={"FrameRate": 30},
            )
            self._still_config = self._picam2.create_still_configuration(
                main={"size": (sw, sh), "format": "RGB888"},
            )

            self._picam2.configure(self._preview_config)
            self._picam2.start()
            self.backend = "picamera2"
            logging.info("Camera backend: Picamera2 (libcamera)")
            return

        except Exception as e:
            logging.warning("Picamera2 not available or failed: %s", e)

        # Fallback OpenCV
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(self.cfg.opencv_device_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.opencv_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.opencv_height)
            ok, _ = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError("OpenCV could not read from camera.")
            self._opencv_cap = cap
            self.backend = "opencv"
            logging.info("Camera backend: OpenCV VideoCapture (USB webcam)")
            return

        except Exception as e:
            logging.error("No camera backend available: %s", e)
            self.backend = "none"

    def available(self) -> bool:
        return self.backend in ("picamera2", "opencv")

    def retry(self) -> None:
        self.close()
        time.sleep(0.4)
        self._init_camera()

    def close(self) -> None:
        try:
            if self._picam2:
                self._picam2.stop()
                self._picam2.close()
        except Exception:
            pass
        self._picam2 = None
        self._preview_config = None
        self._still_config = None

        try:
            if self._opencv_cap is not None:
                self._opencv_cap.release()
        except Exception:
            pass
        self._opencv_cap = None
        self.backend = "none"

    def get_preview_frame(self) -> Optional[np.ndarray]:
        try:
            if self.backend == "picamera2" and self._picam2:
                arr = self._picam2.capture_array("main")
                if self.cfg.picamera2_swap_rb:
                    arr = arr[..., ::-1]
                return arr

            if self.backend == "opencv" and self._opencv_cap is not None:
                import cv2  # type: ignore
                ok, frame = self._opencv_cap.read()
                if not ok:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception:
            return None
        return None

    def capture_still(self) -> Optional[Image.Image]:
        """
        Picamera2 requires camera STOPPED before configure().
        stop -> configure(still) -> start -> capture_array
        stop -> configure(preview) -> start
        """
        try:
            if self.backend == "picamera2" and self._picam2 and self._preview_config and self._still_config:
                self._picam2.stop()
                self._picam2.configure(self._still_config)
                self._picam2.start()

                arr = self._picam2.capture_array("main")
                if self.cfg.picamera2_swap_rb:
                    arr = arr[..., ::-1]

                self._picam2.stop()
                self._picam2.configure(self._preview_config)
                self._picam2.start()

                return Image.fromarray(arr, mode="RGB")

            if self.backend == "opencv" and self._opencv_cap is not None:
                arr = self.get_preview_frame()
                if arr is None:
                    return None
                return Image.fromarray(arr, mode="RGB")

        except Exception as e:
            logging.exception("Still capture failed: %s", e)
            return None
        return None


# ----------------------------
# Filters
# ----------------------------

def apply_filter(img: Image.Image, filter_name: str, brightness: float) -> Image.Image:
    out = img.convert("RGB")

    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    if filter_name == "none":
        return out

    if filter_name == "bw":
        return ImageOps.grayscale(out).convert("RGB")

    if filter_name == "retro":
        out = ImageEnhance.Color(out).enhance(0.75)
        out = ImageEnhance.Contrast(out).enhance(1.15)
        overlay = Image.new("RGB", out.size, (255, 230, 200))
        out = Image.blend(out, overlay, alpha=0.12)
        return out

    return out


# ----------------------------
# Printing
# ----------------------------

class PrintManager:
    def __init__(self, prefer_cups: bool, printer_name: str, copies: int) -> None:
        self.prefer_cups = prefer_cups
        self.printer_name = printer_name
        self.copies = max(1, copies)
        self.lp_path = which("lp")

    def can_print_cups(self) -> bool:
        return self.prefer_cups and (self.lp_path is not None)

    def print_or_queue(self, photo_path: Path) -> Tuple[bool, str]:
        PRINT_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

        if self.can_print_cups():
            try:
                cmd = ["lp", "-n", str(self.copies)]
                if self.printer_name:
                    cmd += ["-d", self.printer_name]
                cmd += [str(photo_path)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True, "Printed via CUPS (lp)"
            except subprocess.CalledProcessError:
                logging.warning("lp failed; falling back to queue")

        try:
            queued = PRINT_QUEUE_DIR / photo_path.name
            shutil.copy2(photo_path, queued)
            return True, f"Queued: {queued.name}"
        except Exception as e:
            logging.exception("Queue copy failed: %s", e)
            return False, "Print/queue failed"


# ----------------------------
# App / UI
# ----------------------------

class PhotoboothApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.running = True

        self.filter_name = cfg.default_filter if cfg.default_filter in cfg.filter_cycle else cfg.filter_cycle[0]
        self.brightness = float(cfg.brightness)
        self.countdown_seconds = int(cfg.countdown_seconds)

        self.last_photo_path: Optional[Path] = None
        self.last_photo_shown_until: float = 0.0

        self.status_message = "Ready"
        self.status_until = 0.0

        self.delete_confirm_mode = False
        self.delete_confirm_until = 0.0

        self.midi_queue: "queue.Queue[MidiEvent]" = queue.Queue()
        self.midi = MidiController(cfg.midi_port_name, self.midi_queue)
        self.midi.start()

        self.camera = CameraController(cfg.camera)
        self.last_camera_ok_time = time.time()

        self.printer = PrintManager(cfg.prefer_cups, cfg.cups_printer_name, cfg.copies)

        # --- Audio: deterministic playback for oscilloscope/vector WAVs ---
        # IMPORTANT: pre_init must happen BEFORE pygame.init(), otherwise SDL may pick a different format.
        self.snd_countdown: Optional[pygame.mixer.Sound] = None
        self.snd_shutter: Optional[pygame.mixer.Sound] = None
        self._ch_countdown: Optional[pygame.mixer.Channel] = None
        self._ch_shutter: Optional[pygame.mixer.Channel] = None

        try:
            # 48kHz is often the cleanest path on Pi/ALSA/HDMI and reduces resample weirdness.
            # If your scope file is authored at 44.1kHz and looks better there, change 48000 -> 44100.
            pygame.mixer.pre_init(48000, -16, 2, 512)
        except Exception:
            pass

        pygame.init()
        pygame.font.init()

        try:
            # Initialize mixer (might already be initialized by pygame.init(), but this is safe)
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.set_num_channels(8)
            self._ch_countdown = pygame.mixer.Channel(0)
            self._ch_shutter = pygame.mixer.Channel(1)

            self.snd_countdown = safe_load_sound(self.cfg.countdown_wav)
            self.snd_shutter = safe_load_sound(self.cfg.shutter_wav)

            # Keep level conservative to avoid "bloom/thick trace" (4171-like look)
            if self.snd_countdown is not None:
                self.snd_countdown.set_volume(0.35)
            if self.snd_shutter is not None:
                self.snd_shutter.set_volume(0.50)

        except Exception:
            logging.exception("Audio init failed (non-fatal).")

        # Display init (robust driver fallback)
        self.screen = init_pygame_display(cfg.fullscreen)
        pygame.display.set_caption(APP_NAME)
        self.w, self.h = self.screen.get_size()

        # Hide cursor in kiosk
        pygame.mouse.set_visible(False)

        # Fonts (bold/techy; safe fallbacks)
        self.font_big = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.14), bold=True)
        self.font_med = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.052), bold=True)
        self.font_small = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.034))
        self.font_hud = pygame.font.SysFont("DejaVu Sans", int(self.h * 0.030), bold=True)

        # Y2K overlays (prebaked for perf)
        self._scanlines = make_scanlines((self.w, self.h), alpha=Y2K["scanline_alpha"], spacing=3)
        self._vignette = make_vignette((self.w, self.h), strength=55)
        self._t0 = time.time()

        # Thumbnail cache (avoid re-opening image every frame)
        self._thumb_cache_path: Optional[Path] = None
        self._thumb_cache_surf: Optional[pygame.Surface] = None

        self.clock = pygame.time.Clock()

    def shutdown(self) -> None:
        self.running = False
        try:
            self.midi.stop()
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass
        pygame.quit()

    def set_status(self, msg: str, seconds: float = 2.0) -> None:
        self.status_message = msg
        self.status_until = time.time() + seconds
        logging.info("STATUS: %s", msg)

    def cycle_filter(self) -> None:
        cycle = list(self.cfg.filter_cycle)
        idx = cycle.index(self.filter_name) if self.filter_name in cycle else 0
        self.filter_name = cycle[(idx + 1) % len(cycle)]
        self.set_status(f"Filter: {self.filter_name}", 1.3)

    def adjust_brightness_from_cc(self, value: int) -> None:
        v = clamp(value / 127.0, 0.0, 1.0)
        self.brightness = 0.5 + v * 1.3
        self.set_status(f"Brightness: {self.brightness:.2f}", 1.0)

    def adjust_countdown_from_cc(self, value: int) -> None:
        v = clamp(value / 127.0, 0.0, 1.0)
        self.countdown_seconds = int(round(1 + v * 9))
        self.set_status(f"Countdown: {self.countdown_seconds}s", 1.0)

    def handle_midi_event(self, ev: MidiEvent) -> None:
        mm = self.cfg.midi_mapping

        if ev.type == "note_on" and ev.note is not None:
            if ev.note == mm.start_note and not self.delete_confirm_mode:
                self.start_photo_sequence(trigger=ev)
            elif ev.note == mm.filter_note and not self.delete_confirm_mode:
                self.cycle_filter()
            elif ev.note == mm.print_note and not self.delete_confirm_mode:
                self.print_last_photo()
            elif ev.note == mm.delete_note:
                self.delete_last_photo_with_confirmation()

        if ev.type == "control_change" and ev.control is not None and ev.value is not None:
            if ev.control == mm.brightness_cc:
                self.adjust_brightness_from_cc(ev.value)
            elif ev.control == mm.countdown_cc:
                self.adjust_countdown_from_cc(ev.value)

    def delete_last_photo_with_confirmation(self) -> None:
        if not self.last_photo_path or not self.last_photo_path.exists():
            if not self.cfg.minimal_ui:
                self.set_status("No photo to delete", 2.0)
            return

        now = time.time()
        if not self.delete_confirm_mode:
            self.delete_confirm_mode = True
            self.delete_confirm_until = now + 5.0
            if not self.cfg.minimal_ui:
                self.set_status("Press DELETE again to confirm", 5.0)
            return

        if now <= self.delete_confirm_until:
            try:
                p = self.last_photo_path
                meta = p.with_suffix(".json")
                p.unlink(missing_ok=True)      # type: ignore[arg-type]
                meta.unlink(missing_ok=True)   # type: ignore[arg-type]
                self.last_photo_path = None
                self._thumb_cache_path = None
                self._thumb_cache_surf = None
                if not self.cfg.minimal_ui:
                    self.set_status("Deleted last photo", 2.0)
            except Exception as e:
                logging.exception("Delete failed: %s", e)
                if not self.cfg.minimal_ui:
                    self.set_status("Delete failed", 2.5)
        else:
            if not self.cfg.minimal_ui:
                self.set_status("Delete confirmation timed out", 2.0)

        self.delete_confirm_mode = False

    def print_last_photo(self) -> None:
        if not self.last_photo_path or not self.last_photo_path.exists():
            if not self.cfg.minimal_ui:
                self.set_status("No photo to print", 2.0)
            return
        ok, msg = self.printer.print_or_queue(self.last_photo_path)
        if not self.cfg.minimal_ui:
            self.set_status(msg if ok else "Print failed", 2.5)

    def start_photo_sequence(self, trigger: MidiEvent) -> None:
        if not self.camera.available():
            if not self.cfg.minimal_ui:
                self.set_status("Camera unavailable, retrying…", 2.0)
            self.camera.retry()
            return

        # Countdown
        for t in range(self.countdown_seconds, 0, -1):
            try:
                if self.snd_countdown is not None:
                    # Dedicated channel + stop first => avoids overlapping ticks (smeared XY traces)
                    if hasattr(self, '_ch_countdown') and self._ch_countdown is not None:
                        self._ch_countdown.stop()
                        self._ch_countdown.play(self.snd_countdown)
                    else:
                        self.snd_countdown.play()
            except Exception:
                pass

            self._draw_frame(countdown=t)
            pygame.display.flip()
            self.clock.tick(60)
            time.sleep(1.0)

        # Optional shutter sound
        try:
            if self.snd_shutter is not None:
                if hasattr(self, '_ch_shutter') and self._ch_shutter is not None:
                    self._ch_shutter.stop()
                    self._ch_shutter.play(self.snd_shutter)
                else:
                    self.snd_shutter.play()
        except Exception:
            pass

        img = self.camera.capture_still()
        if img is None:
            if not self.cfg.minimal_ui:
                self.set_status("Capture failed (retrying camera)", 3.0)
            self.camera.retry()
            return

        dt = now_dt()
        day_dir = PHOTOS_DIR / today_folder_name(dt)
        day_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{timestamp_name(dt)}_{self.filter_name}.jpg"
        photo_path = day_dir / fname

        try:
            processed = apply_filter(img, self.filter_name, self.brightness)
            processed.save(photo_path, format="JPEG", quality=95)

            meta = {
                "timestamp_iso": dt.isoformat(),
                "photo_path": str(photo_path),
                "filter": self.filter_name,
                "brightness": self.brightness,
                "countdown_seconds": self.countdown_seconds,
                "camera_backend": self.camera.backend,
                "camera_picamera2_swap_rb": self.cfg.camera.picamera2_swap_rb,
                "midi_trigger": trigger.raw if trigger.raw else {},
            }
            safe_write_json(photo_path.with_suffix(".json"), meta)

            self.last_photo_path = photo_path
            self.last_photo_shown_until = time.time() + float(self.cfg.last_photo_show_seconds)

            self._thumb_cache_path = None
            self._thumb_cache_surf = None

            if not self.cfg.minimal_ui:
                self.set_status("Saved!", 1.5)

        except Exception as e:
            logging.exception("Save failed: %s", e)
            if not self.cfg.minimal_ui:
                self.set_status("Save failed", 3.0)

    # --- drawing helpers ---

    def _rotate_if_needed(self, rgb: np.ndarray) -> np.ndarray:
        deg = self.cfg.rotate_preview_degrees % 360
        if deg == 0:
            return rgb
        if deg == 90:
            return np.rot90(rgb, k=3)
        if deg == 180:
            return np.rot90(rgb, k=2)
        if deg == 270:
            return np.rot90(rgb, k=1)
        return rgb

    def _apply_preview_brightness(self, rgb: np.ndarray) -> np.ndarray:
        if abs(self.brightness - 1.0) < 0.01:
            return rgb
        arr = rgb.astype(np.float32) * float(self.brightness)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _ensure_thumb_cache(self, max_size: tuple[int, int]) -> None:
        if not self.last_photo_path or not self.last_photo_path.exists():
            self._thumb_cache_path = None
            self._thumb_cache_surf = None
            return
        if self._thumb_cache_path == self.last_photo_path and self._thumb_cache_surf is not None:
            return
        try:
            thumb = Image.open(self.last_photo_path).convert("RGB")
            thumb = ImageOps.contain(thumb, max_size)
            arr = np.array(thumb)
            self._thumb_cache_surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            self._thumb_cache_path = self.last_photo_path
        except Exception:
            self._thumb_cache_path = None
            self._thumb_cache_surf = None

    def _draw_frame(self, countdown: Optional[int] = None) -> None:
        self.screen.fill((0, 0, 0))

        # Fullscreen last-photo display for a few seconds
        if self.last_photo_path and time.time() < self.last_photo_shown_until and self.last_photo_path.exists():
            try:
                img = Image.open(self.last_photo_path).convert("RGB")
                img = ImageOps.contain(img, (self.w, self.h))
                arr = np.array(img)
                surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                rect = surf.get_rect(center=(self.w // 2, self.h // 2))
                self.screen.blit(surf, rect)

                # No label text in minimal mode
                if not self.cfg.minimal_ui:
                    pad = max(10, int(self.w * 0.012))
                    r = max(10, int(self.w * 0.016))
                    stroke = max(2, int(self.w * 0.0025))
                    label_h = max(52, int(self.h * 0.11))
                    label_rect = pygame.Rect(int(self.w * 0.18), int(self.h * 0.06), int(self.w * 0.64), label_h)
                    draw_bevel_panel(self.screen, label_rect, (10, 12, 18, 160), Y2K["panel_stroke_hot"], r, stroke, 2)
                    draw_glow_text(
                        self.screen,
                        "LAST PHOTO",
                        self.font_med,
                        (label_rect.centerx, label_rect.centery),
                        color=Y2K["text"],
                        glow_color=Y2K["accent_magenta"],
                        center=True,
                        glow_px=6,
                        glow_layers=6,
                        alpha_max=170,
                    )

                self.screen.blit(self._scanlines, (0, 0))
                self.screen.blit(self._vignette, (0, 0))
                return
            except Exception:
                pass

        # Live preview
        frame = self.camera.get_preview_frame() if self.camera.available() else None
        if frame is None:
            if time.time() - self.last_camera_ok_time > 2.0:
                self.camera.retry()
                self.last_camera_ok_time = time.time()

            # No error text in minimal mode
            if not self.cfg.minimal_ui:
                draw_glow_text(
                    self.screen,
                    "CAMERA ERROR",
                    self.font_big,
                    (self.w // 2, int(self.h * 0.38)),
                    color=(255, 200, 200),
                    glow_color=(255, 80, 80),
                    center=True,
                    glow_px=10,
                    glow_layers=7,
                    alpha_max=190,
                )
                draw_glow_text(
                    self.screen,
                    "Retrying…",
                    self.font_med,
                    (self.w // 2, int(self.h * 0.50)),
                    color=Y2K["text"],
                    glow_color=Y2K["accent_cyan"],
                    center=True,
                    glow_px=4,
                    glow_layers=5,
                    alpha_max=120,
                )
        else:
            self.last_camera_ok_time = time.time()
            frame = self._rotate_if_needed(frame)
            frame = self._apply_preview_brightness(frame)

            img = Image.fromarray(frame, mode="RGB")
            img = ImageOps.contain(img, (self.w, self.h))
            arr = np.array(img)
            surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            rect = surf.get_rect(center=(self.w // 2, self.h // 2))
            self.screen.blit(surf, rect)

        # Preview-only mode: NO HUD TEXT/PANELS (your request)
        if self.cfg.minimal_ui:
            self.screen.blit(self._scanlines, (0, 0))
            self.screen.blit(self._vignette, (0, 0))
            return

        # --- Full Y2K HUD overlays (only when minimal_ui is false) ---
        t = time.time()
        pulse = 0.5 + 0.5 * math.sin((t - self._t0) * 2.8)
        pulse2 = 0.5 + 0.5 * math.sin((t - self._t0) * 5.2)

        pad = max(10, int(self.w * 0.012))
        r = max(10, int(self.w * 0.016))
        stroke = max(2, int(self.w * 0.0025))

        # TOP HUD
        top_h = max(56, int(self.h * 0.12))
        top_rect = pygame.Rect(pad, pad, self.w - pad * 2, top_h)
        draw_bevel_panel(self.screen, top_rect, Y2K["panel_fill"], Y2K["panel_stroke"], r, stroke, 2)

        draw_glow_text(
            self.screen,
            APP_NAME,
            self.font_med,
            (top_rect.left + pad, top_rect.centery - int(top_rect.h * 0.12)),
            color=Y2K["text"],
            glow_color=Y2K["accent_cyan"],
            center=False,
            glow_px=7,
            glow_layers=6,
            alpha_max=int(120 + 80 * pulse),
        )

        chip_w = max(180, int(self.w * 0.25))
        chip_h = max(40, int(top_rect.h * 0.55))
        chip_rect = pygame.Rect(top_rect.right - pad - chip_w, top_rect.centery - chip_h // 2, chip_w, chip_h)
        draw_bevel_panel(self.screen, chip_rect, Y2K["panel_fill_2"], Y2K["panel_stroke_hot"], max(10, r - 4), stroke, 2)
        draw_glow_text(
            self.screen,
            f"FILTER · {self.filter_name.upper()}",
            self.font_hud,
            (chip_rect.centerx, chip_rect.centery),
            color=Y2K["text"],
            glow_color=Y2K["accent_magenta"],
            center=True,
            glow_px=4,
            glow_layers=5,
            alpha_max=int(110 + 70 * pulse2),
        )

        mini = f"BRI {self.brightness:.2f}   ·   COUNTDOWN {self.countdown_seconds}s"
        draw_glow_text(
            self.screen,
            mini,
            self.font_small,
            (top_rect.left + pad, top_rect.bottom - pad - int(self.font_small.get_height() * 0.9)),
            color=Y2K["text_dim"],
            glow_color=Y2K["accent_lime"],
            center=False,
            glow_px=2,
            glow_layers=4,
            alpha_max=int(70 + 40 * pulse),
        )

        # CENTER countdown digits
        if countdown is not None:
            box_w = max(180, int(self.w * 0.22))
            box_h = max(180, int(self.h * 0.30))
            box = pygame.Rect(self.w // 2 - box_w // 2, int(self.h * 0.42) - box_h // 2, box_w, box_h)
            draw_bevel_panel(
                self.screen,
                box,
                (8, 10, 16, 120),
                (60, 255, 220, int(120 + 60 * pulse)),
                max(14, r),
                stroke,
                2,
            )
            draw_glow_text(
                self.screen,
                str(countdown),
                self.font_big,
                (self.w // 2, int(self.h * 0.42)),
                color=Y2K["text"],
                glow_color=Y2K["accent_cyan"],
                center=True,
                glow_px=10,
                glow_layers=7,
                alpha_max=int(130 + 90 * pulse2),
            )

        # CRT overlays last
        self.screen.blit(self._scanlines, (0, 0))
        self.screen.blit(self._vignette, (0, 0))

    def run(self) -> None:
        if not self.cfg.minimal_ui:
            self.set_status("Ready", 1.0)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # emergency exits
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.start_photo_sequence(trigger=MidiEvent(type="manual", raw={"type": "manual"}))
                    elif event.key == pygame.K_f:
                        self.cycle_filter()

            # drain MIDI queue
            try:
                while True:
                    ev = self.midi_queue.get_nowait()
                    self.handle_midi_event(ev)
            except queue.Empty:
                pass

            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(30)

        self.shutdown()


# ----------------------------
# Main
# ----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def print_midi_ports_and_exit() -> None:
    ports = MidiController.list_ports()
    print("Available MIDI input ports:")
    for p in ports:
        print(f" - {p}")
    if not ports:
        print(" (none found)")
    sys.exit(0)


def main() -> None:
    setup_logging()
    ensure_dirs()

    if "--list-midi" in sys.argv:
        print_midi_ports_and_exit()

    auto_set_sdl_videodriver()
    logging.info(
        "DISPLAY=%s WAYLAND_DISPLAY=%s SDL_VIDEODRIVER=%s",
        os.environ.get("DISPLAY"),
        os.environ.get("WAYLAND_DISPLAY"),
        os.environ.get("SDL_VIDEODRIVER"),
    )

    cfg = load_or_create_config(DEFAULT_CONFIG_PATH)

    logging.info("Starting %s", APP_NAME)
    logging.info("Config (preferred): %s", (DEFAULT_INI_PATH if DEFAULT_INI_PATH.exists() else DEFAULT_JSON_PATH).resolve())
    logging.info("Photos dir: %s", PHOTOS_DIR.resolve())
    logging.info("Print queue dir: %s", PRINT_QUEUE_DIR.resolve())
    logging.info("Tip: list MIDI ports with: .venv/bin/python photobooth.py --list-midi")

    app = PhotoboothApp(cfg)
    try:
        logging.info("pygame video driver selected: %s", pygame.display.get_driver())
        app.run()
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    main()
