from escpos.printer import Usb as _BaseUsb
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
VENDOR_ID = 0x0525
PRODUCT_ID = 0xA700

# 80mm paper = 576 dots wide at 203 DPI; 72mm = 512 dots; 58mm = 384 dots
PRINTER_WIDTH = 576

BRAND_TOP = "3615 TV STORE"
BRAND_SUB = "PHOTOBOOTH // ANALOG DREAMS"
FOOTER_1 = "tvstore.fr"
FOOTER_2 = "Merci • Keep the signal alive"
QR_URL = "https://tvstore.fr"


# -----------------------------
# USB BACKEND — chunked writes
# -----------------------------

class _ChunkedUsb(_BaseUsb):
    """python-escpos Usb backend with chunked writes.

    The DS-920's USB receive buffer (~4-8 KB) overflows when the host sends
    the full image (~30 KB) in one bulk transfer, causing the printer to stall
    the endpoint ([Errno 32] Pipe error). Sending in 4 KB chunks with a small
    delay between each lets the printer drain its buffer while receiving.
    """
    _CHUNK = 2048   # bytes per USB write call
    _DELAY = 0.40   # seconds between chunks (printer drain rate ~15-24 KB/s under load)
    # Maths: send rate = _CHUNK / _DELAY = 2048/0.30 ≈ 6.8 KB/s
    # At 40mm/s print speed × 8 dots/mm × 72 bytes/line ≈ 23 KB/s drain rate.
    # Under heavy thermal load (57% black) the effective drain rate drops; keeping
    # send rate well below drain rate prevents the ~8 KB USB receive buffer from
    # filling and causing an endpoint stall ([Errno 32] Pipe error).

    def __init__(self, vendor_id, product_id):
        import usb.core
        # Must be initialised before super().__init__() in case it calls _raw().
        self._bytes_since_delay = 0
        # Detach ALL kernel interfaces before super().__init__ tries to open.
        # Requires either root OR udev MODE="0666" on the device (see setup instructions).
        dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if dev is not None:
            for cfg in dev:
                for intf in cfg:
                    ifnum = intf.bInterfaceNumber
                    try:
                        if dev.is_kernel_driver_active(ifnum):
                            dev.detach_kernel_driver(ifnum)
                            logging.warning("[USB] Detached kernel driver from interface %d", ifnum)
                        else:
                            logging.warning("[USB] Interface %d: no kernel driver active", ifnum)
                    except Exception as e:
                        logging.warning("[USB] Detach iface %d FAILED: %s", ifnum, e)
            # Do NOT reset the device here — reset causes the kernel to immediately
            # re-bind its drivers before super().__init__ can claim the interfaces.
        super().__init__(vendor_id, product_id)
        # Clear any stale bulk-OUT endpoint halt left over from a previous
        # failed print.  On Linux, libusb skips SET_CONFIGURATION when the
        # device is already in that configuration, so the STALL condition is
        # never reset and the very first write of the next job fails with
        # [Errno 32] Pipe error.  An explicit CLEAR_FEATURE(ENDPOINT_HALT)
        # control transfer fixes this without requiring a full device reset.
        try:
            self.device.clear_halt(self.out_ep)
            logging.warning("[USB] Cleared endpoint halt on out_ep=0x%02x", self.out_ep)
        except Exception as e:
            logging.warning("[USB] clear_halt(out_ep): %s", e)

    def _raw(self, msg: bytes) -> None:
        """Rate-limit USB output to prevent printer buffer overflow.

        Tracks cumulative bytes across ALL calls so that patterns of many
        small _raw() calls (e.g. one per image row in some escpos versions)
        are throttled the same as a single large call.
        """
        offset = 0
        while offset < len(msg):
            # How many bytes can we send before we must pause?
            headroom = self._CHUNK - self._bytes_since_delay
            if headroom <= 0:
                time.sleep(self._DELAY)
                self._bytes_since_delay = 0
                headroom = self._CHUNK
            chunk = msg[offset:offset + headroom]
            self.device.write(self.out_ep, chunk, self.timeout)
            sent = len(chunk)
            offset += sent
            self._bytes_since_delay += sent
            # Hit the threshold — pause before next chunk.
            # Intentionally NO "and offset < len(msg)" so the sleep also
            # fires at the end of a message, throttling the next _raw() call.
            if self._bytes_since_delay >= self._CHUNK:
                time.sleep(self._DELAY)
                self._bytes_since_delay = 0


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

def prep_photo(path: str, target_w: int) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = ImageOps.crop(img, border=10)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Brightness(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.2)
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
    qr_w = int(target_w * 0.55)
    img = img.resize((qr_w, qr_w), Image.NEAREST)
    pad_left = (target_w - qr_w) // 2
    pad_right = target_w - qr_w - pad_left
    img = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=1)
    return img


# -----------------------------
# PRINT
# -----------------------------

def print_receipt(photo_path: str, frame_id: str | None = None):
    if not os.path.exists(photo_path):
        raise FileNotFoundError(photo_path)

    now = dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d  %H:%M:%S")
    if frame_id is None:
        frame_id = now.strftime("%y%m%d-%H%M%S")

    # _ChunkedUsb.__init__ calls detach_kernel_driver() directly via pyusb.
    # This works without root when the udev rule sets MODE="0666" on the
    # printer's USB device node.  No need for sysfs unbind or rmmod.
    logging.warning("[PRINT] Opening printer USB %04x:%04x", VENDOR_ID, PRODUCT_ID)
    p = _ChunkedUsb(VENDOR_ID, PRODUCT_ID)

    # Header
    p.set(align="center", bold=True, width=2, height=2)
    p.text(BRAND_TOP + "\n")
    p.set(align="center", bold=False, width=1, height=1)
    p.text(BRAND_SUB + "\n")
    p.text("-" * 32 + "\n")

    # Meta
    p.set(align="center")
    p.text(f"{stamp}\n")
    p.text(f"FRAME: {frame_id}\n")
    p.text("\n")

    # Photo
    photo = prep_photo(photo_path, PRINTER_WIDTH)
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
    p.image(photo)
    p.text("\n")

    # QR
    if QR_URL:
        p.text("SCAN / DOWNLOAD\n")
        p.image(make_qr(QR_URL, PRINTER_WIDTH))
        p.text("\n")

    # Footer
    p.set(align="center", bold=True)
    p.text(FOOTER_1 + "\n")
    p.set(align="center", bold=False)
    p.text(FOOTER_2 + "\n")
    p.text("\n\n")
    p.cut()
    logging.info("Receipt done.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    # Sanity-check the image before printing
    img = prep_photo(path, PRINTER_WIDTH)
    pixels = list(img.getdata())
    black = sum(1 for px in pixels if px == 0)
    total = len(pixels)
    pct = 100 * black // total if total else 0
    print(f"Image: {img.size[0]}×{img.size[1]} px, {black}/{total} black dots ({pct}%)")
    if pct < 1:
        print("WARNING: image is nearly all white — check brightness/exposure of source photo!")

    print_receipt(path)
