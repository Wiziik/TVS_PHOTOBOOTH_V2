from escpos.printer import Usb as _BaseUsb, File as _BaseFile
from PIL import Image, ImageOps, ImageEnhance
import datetime as dt
import logging
import os
import subprocess
import time
import qrcode

# -----------------------------
# CONFIG
# -----------------------------
VENDOR_ID = 0x04B8
PRODUCT_ID = 0x0E15

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
}

RECEIPT_TEXT_TEMPLATE = """# Edit these values to customize printed text.
# Format: KEY = value
# Leave QR_URL empty to disable QR code printing.

BRAND_TOP = 3615 TV STORE
BRAND_SUB = PHOTOBOOTH // ANALOG DREAMS
FOOTER_1 = tvstore.fr
FOOTER_2 = Merci • Keep the signal alive
QR_URL = https://tvstore.fr
FRAME_LABEL = FRAME
SCAN_LABEL = SCAN / DOWNLOAD
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

    def __init__(self, vendor_id, product_id):
        import usb.core
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


def _unbind_printer_from_kernel():
    """Surgically unbind kernel drivers (usblp, cdc_acm) from ONLY the printer's
    USB interfaces using sysfs.  This leaves every other device (e.g. Arduino
    Leonardo MIDI) completely untouched — unlike a global rmmod."""
    import glob
    VID_STR = f"{VENDOR_ID:04x}"
    PID_STR = f"{PRODUCT_ID:04x}"
    for driver in ("usblp", "cdc_acm"):
        pattern = f"/sys/bus/usb/drivers/{driver}/*"
        for path in glob.glob(pattern):
            if not os.path.isdir(path):
                continue
            # Walk up to the USB device to read VID/PID
            for rel in ("../../", "../"):
                try:
                    vid = open(os.path.join(path, rel, "idVendor")).read().strip()
                    pid = open(os.path.join(path, rel, "idProduct")).read().strip()
                    break
                except OSError:
                    vid = pid = ""
            if vid != VID_STR or pid != PID_STR:
                continue
            iface = os.path.basename(path)
            try:
                with open(f"/sys/bus/usb/drivers/{driver}/unbind", "w") as f:
                    f.write(iface)
                logging.info("Unbound %s from %s", iface, driver)
            except OSError as e:
                # Need root — fall back to sudo rmmod for usblp only (safe)
                logging.debug("sysfs unbind %s/%s: %s", driver, iface, e)
                if driver == "usblp":
                    try:
                        subprocess.run(
                            ["sudo", "rmmod", "usblp"],
                            capture_output=True, timeout=3
                        )
                    except Exception:
                        pass


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
    photo_path: str,
    frame_id: str | None = None,
    reduce_factor: float = 1.0,
    qr_url: str | None = None,
    brightness: float = 1.3,
    contrast: float = 1.1,
):
    if not os.path.exists(photo_path):
        raise FileNotFoundError(photo_path)

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

    logging.warning("[PRINT] Opening printer USB %04x:%04x", VENDOR_ID, PRODUCT_ID)
    logging.warning(
        "[PRINT] Width reduce_factor=%.2f -> target_width=%d (base=%d)",
        reduce_factor, target_width, PRINTER_WIDTH
    )
    p = _ChunkedUsb(VENDOR_ID, PRODUCT_ID)
    try:
        # Set lower print density to prevent thermal stalls.
        # GS 7 n1 n2: n1=heating dots (1-255), n2=heating time (3-15)
        # Lower values = less heat = fewer stalls. Default is ~9,80.
        p._raw(b'\x1d\x37\x04\x40')  # 4 heating dots, 64 heating time
        # Set print speed to medium to reduce thermal load
        p._raw(b'\x1d\x28\x4b\x02\x00\x31\x02')  # speed level 2 (slower)

        # Header
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

        # Photo
        photo = prep_photo(photo_path, target_width, brightness=brightness, contrast=contrast)
        pixels = list(photo.getdata())
        black = sum(1 for px in pixels if px == 0)
        total = len(pixels)
        pct = 100 * black // total if total else 0
        logging.warning(
            "[PRINT] Photo: %dx%d px — %d/%d black dots (%d%%)",
            photo.size[0], photo.size[1], black, total, pct,
        )
        if pct < 2:
            logging.warning(
                "[PRINT] Image is nearly all white (%d%% black) — source photo may be blank!", pct
            )
        p.image(photo, impl="graphics", fragment_height=128)
        p.text("\n")

        # QR
        if text_cfg["QR_URL"]:
            if text_cfg["SCAN_LABEL"]:
                p.text(text_cfg["SCAN_LABEL"] + "\n")
            p.image(make_qr(text_cfg["QR_URL"], target_width))
            p.text("\n")

        # Footer
        p.set(align="center", text_type="B")
        p.text(text_cfg["FOOTER_1"] + "\n")
        p.set(align="center", text_type="normal")
        p.text(text_cfg["FOOTER_2"] + "\n")
        p.text("\n\n")
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
