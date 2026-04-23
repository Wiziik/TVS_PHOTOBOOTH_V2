from escpos.printer import Usb as _BaseUsb
from PIL import Image, ImageOps, ImageEnhance
import datetime as dt
import logging
import os
import time
import qrcode

# -----------------------------
# CONFIG
# -----------------------------
# Fallback only — by default the driver auto-detects the first USB device
# that exposes a printer-class interface (bInterfaceClass == 0x07).
VENDOR_ID = 0x0525
PRODUCT_ID = 0xA700


def _find_first_usb_printer() -> tuple[int, int]:
    """Return (vid, pid) of the first USB device exposing a printer-class interface."""
    import usb.core
    for dev in usb.core.find(find_all=True) or []:
        try:
            for cfg in dev:
                for intf in cfg:
                    if intf.bInterfaceClass == 0x07:
                        return int(dev.idVendor), int(dev.idProduct)
        except Exception:
            continue
    raise RuntimeError("No USB printer-class device found.")

# 80mm paper = 576 dots wide at 203 DPI; 72mm = 512 dots; 58mm = 384 dots
PRINTER_WIDTH = 576

BRAND_TOP = "3615 TV STORE"
BRAND_SUB = "PHOTOBOOTH // ANALOG DREAMS"
FOOTER_1 = "tvstore.fr"
FOOTER_2 = "Merci • Keep the signal alive"
QR_URL = "https://tvstore.fr"
FRAME_LABEL = "FRAME"
SCAN_LABEL = "SCAN / DOWNLOAD"

RECEIPT_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "receipt_text.txt",
)

RECEIPT_TEXT_DEFAULTS = {
    "BRAND_TOP": BRAND_TOP,
    "BRAND_SUB": BRAND_SUB,
    "FOOTER_1": FOOTER_1,
    "FOOTER_2": FOOTER_2,
    "QR_URL": QR_URL,
    "FRAME_LABEL": FRAME_LABEL,
    "SCAN_LABEL": SCAN_LABEL,
    "END_IMAGE": "",
}

RECEIPT_TEXT_TEMPLATE = """# Edit these values to customize printed text.
# Format: KEY = value
# Leave QR_URL empty to disable QR code printing.
# Leave END_IMAGE empty to skip the end-of-receipt image.

BRAND_TOP = 3615 TV STORE
BRAND_SUB = PHOTOBOOTH // ANALOG DREAMS
FOOTER_1 = tvstore.fr
FOOTER_2 = Merci • Keep the signal alive
QR_URL = https://tvstore.fr
FRAME_LABEL = FRAME
SCAN_LABEL = Scan pour récupérer tes photos en ligne.
END_IMAGE = ./Do_not_throw_logo.png
"""


def _ensure_receipt_text_file(path: str = RECEIPT_TEXT_PATH) -> None:
    if os.path.exists(path):
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(RECEIPT_TEXT_TEMPLATE)
        logging.info("Created default receipt text file: %s", path)
    except Exception as e:
        logging.warning("Could not create receipt text file %s: %s", path, e)


def load_receipt_text(path: str = RECEIPT_TEXT_PATH) -> dict[str, str]:
    text = dict(RECEIPT_TEXT_DEFAULTS)
    _ensure_receipt_text_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logging.warning("[TEXT] Ignoring malformed line %d: %r", lineno, raw.rstrip("\n"))
                    continue
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip()
                if key in text:
                    text[key] = value
                else:
                    logging.warning("[TEXT] Unknown key %r on line %d", key, lineno)
    except Exception as e:
        logging.warning("Could not read receipt text file %s: %s", path, e)
    return text


# -----------------------------
# USB BACKEND — chunked writes
# -----------------------------

class _ChunkedUsb(_BaseUsb):
    """python-escpos Usb backend with chunked writes and delays
    to prevent the printer's receive buffer from overflowing."""
    _CHUNK = 4096
    _DELAY = 0.10

    def __init__(self, vendor_id=None, product_id=None):
        import usb.core
        if vendor_id is None or product_id is None:
            try:
                vendor_id, product_id = _find_first_usb_printer()
                logging.warning("[PRINT] Auto-detected USB printer %04x:%04x",
                                vendor_id, product_id)
            except RuntimeError:
                vendor_id, product_id = VENDOR_ID, PRODUCT_ID
                logging.warning("[PRINT] Auto-detect failed; falling back to %04x:%04x",
                                vendor_id, product_id)
        dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if dev is not None:
            for cfg in dev:
                for intf in cfg:
                    ifnum = intf.bInterfaceNumber
                    try:
                        if dev.is_kernel_driver_active(ifnum):
                            dev.detach_kernel_driver(ifnum)
                    except Exception:
                        pass
        super().__init__(vendor_id, product_id)
        try:
            self.device.clear_halt(self.out_ep)
        except Exception:
            pass

    def _raw(self, msg: bytes) -> None:
        offset = 0
        while offset < len(msg):
            chunk = msg[offset:offset + self._CHUNK]
            self.device.write(self.out_ep, chunk, self.timeout)
            offset += len(chunk)
            if offset < len(msg):
                time.sleep(self._DELAY)


# -----------------------------
# IMAGE HELPERS
# -----------------------------

def prep_photo(path: str, target_w: int, brightness: float = 1.3, contrast: float = 1.1) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = ImageOps.crop(img, border=5)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    wpercent = target_w / float(img.size[0])
    hsize = int(float(img.size[1]) * wpercent)
    img = img.resize((target_w, hsize), Image.LANCZOS)
    img = img.convert("1")
    return img


def prep_end_image(path: str, target_w: int, scale: float = 0.25) -> Image.Image:
    """Load a small icon for the end of the receipt, fit to scale × printer width."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    # Flatten transparency against white so it prints cleanly
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = bg
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    dest_w = max(48, int(target_w * scale))
    wpercent = dest_w / float(img.size[0])
    dest_h = max(48, int(img.size[1] * wpercent))
    img = img.resize((dest_w, dest_h), Image.LANCZOS)
    img = img.convert("1")
    pad_left = (target_w - dest_w) // 2
    pad_right = target_w - dest_w - pad_left
    return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=1)


def make_qr(url: str, target_w: int) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("1")
    qr_w = int(target_w * 0.40)
    img = img.resize((qr_w, qr_w), Image.NEAREST)
    pad_left = (target_w - qr_w) // 2
    pad_right = target_w - qr_w - pad_left
    img = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=1)
    return img


# -----------------------------
# PRINT
# -----------------------------

def print_receipt(
    photo_path,
    frame_id: str | None = None,
    reduce_factor: float = 1.0,
    qr_url: str | None = None,
    brightness: float = 1.3,
    contrast: float = 1.1,
):
    # Accept a single path or a list — normalise to a list.
    if isinstance(photo_path, (str, bytes, os.PathLike)):
        photo_paths = [os.fspath(photo_path)]
    else:
        photo_paths = [os.fspath(p) for p in photo_path]
    for pp in photo_paths:
        if not os.path.exists(pp):
            raise FileNotFoundError(pp)

    now = dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d  %H:%M:%S")
    if frame_id is None:
        frame_id = now.strftime("%y%m%d-%H%M%S")

    # Clamp scale: 1.0 = full width, 0.8 = 20% narrower.
    try:
        reduce_factor = float(reduce_factor)
    except Exception:
        reduce_factor = 1.0
    reduce_factor = max(0.1, min(1.0, reduce_factor))
    target_width = max(64, int(round(PRINTER_WIDTH * reduce_factor)))
    text_cfg = load_receipt_text()
    if qr_url is not None:
        text_cfg["QR_URL"] = str(qr_url).strip()

    logging.warning(
        "[PRINT] Width reduce_factor=%.2f -> target_width=%d (base=%d) | %d photo(s)",
        reduce_factor, target_width, PRINTER_WIDTH, len(photo_paths),
    )
    p = _ChunkedUsb()
    try:
        # ESC @ — reset/init. Widely supported; printers that don't know
        # Epson-specific density/speed prelude would print those as garbage.
        p._raw(b'\x1b\x40')

        # Header (printed once per strip)
        p.set(align="center", text_type="B", width=2, height=2)
        p.text(text_cfg["BRAND_TOP"] + "\n")
        p.set(align="center", text_type="normal", width=1, height=1)
        p.text(text_cfg["BRAND_SUB"] + "\n")
        p.text("-" * 32 + "\n")

        # Meta
        p.set(align="center")
        p.text(f"{stamp}\n")
        p.text(f"{text_cfg['FRAME_LABEL']}: {frame_id}\n")
        p.text("\n")

        # Photos stacked in order
        for idx, pp in enumerate(photo_paths, start=1):
            photo = prep_photo(pp, target_width, brightness=brightness, contrast=contrast)
            pixels = list(photo.getdata())
            black = sum(1 for px in pixels if px == 0)
            total = len(pixels)
            pct = 100 * black // total if total else 0
            logging.warning(
                "[PRINT] Photo %d/%d: %dx%d px — %d%% black",
                idx, len(photo_paths), photo.size[0], photo.size[1], pct,
            )
            if pct < 2:
                logging.warning(
                    "[PRINT] Photo %d is nearly all white (%d%% black) — source may be blank!",
                    idx, pct,
                )
            p.image(photo, impl="bitImageRaster", fragment_height=128)
            # Small gap between photos on the strip
            if idx < len(photo_paths):
                p.text("\n")

        p.text("\n")

        # QR (once, after all photos)
        if text_cfg["QR_URL"]:
            if text_cfg["SCAN_LABEL"]:
                p.text(text_cfg["SCAN_LABEL"] + "\n")
            p.image(make_qr(text_cfg["QR_URL"], target_width), impl="bitImageRaster")
            p.text("\n")

        # Footer
        p.set(align="center", text_type="B")
        p.text(text_cfg["FOOTER_1"] + "\n")
        p.set(align="center", text_type="normal")
        p.text(text_cfg["FOOTER_2"] + "\n")
        p.text("\n")

        # End-of-receipt icon (e.g. "do not throw away")
        end_image_path = text_cfg.get("END_IMAGE", "").strip()
        if end_image_path:
            if not os.path.isabs(end_image_path):
                end_image_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), end_image_path
                )
            if os.path.exists(end_image_path):
                try:
                    end_img = prep_end_image(end_image_path, target_width)
                    p.image(end_img, impl="bitImageRaster")
                    p.text("\n")
                except Exception as e:
                    logging.warning("[PRINT] Failed to print END_IMAGE %s: %s",
                                    end_image_path, e)
            else:
                logging.warning("[PRINT] END_IMAGE not found: %s", end_image_path)

        p.text("\n")
        p.cut()
        logging.info("Receipt done.")
    finally:
        try:
            p.close()
            logging.warning("[PRINT] Closed printer handle")
        except Exception as e:
            logging.warning("[PRINT] close(): %s", e)
        try:
            import usb.util
            dev = getattr(p, "device", None)
            if dev is not None:
                usb.util.dispose_resources(dev)
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    reduce_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    qr_url = sys.argv[3] if len(sys.argv) > 3 else None

    # Sanity-check the image before printing
    target_w = max(64, int(round(PRINTER_WIDTH * max(0.1, min(1.0, reduce_factor)))))
    img = prep_photo(path, target_w)
    pixels = list(img.getdata())
    black = sum(1 for px in pixels if px == 0)
    total = len(pixels)
    pct = 100 * black // total if total else 0
    print(f"Image: {img.size[0]}×{img.size[1]} px, {black}/{total} black dots ({pct}%)")
    if pct < 1:
        print("WARNING: image is nearly all white — check brightness/exposure of source photo!")

    print_receipt(path, reduce_factor=reduce_factor, qr_url=qr_url)
