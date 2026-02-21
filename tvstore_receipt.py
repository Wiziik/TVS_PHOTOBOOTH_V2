from escpos.printer import Usb
from PIL import Image, ImageOps, ImageEnhance
import datetime as dt
import os
import qrcode

# -----------------------------
# CONFIG
# -----------------------------
VENDOR_ID = 0x0525
PRODUCT_ID = 0xA700

# Set paper width:
# 58mm printers are usually 384 dots wide, 80mm are usually 576.
PRINTER_WIDTH = 576  # change to 576 if 80mm

BRAND_TOP = "3615 TV STORE"
BRAND_SUB = "PHOTOBOOTH // ANALOG DREAMS"
FOOTER_1 = "tvstore.fr"
FOOTER_2 = "Merci • Keep the signal alive"

# If you want a QR code on receipts (link to download page, insta, etc.)
QR_URL = "https://tvstore.fr"  # change to your gallery link when ready

# -----------------------------
# IMAGE HELPERS
# -----------------------------
def prep_photo(path: str, target_w: int) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    # Optional: crop a bit to make subject bigger (tune 0..40)
    # border=0 means no zoom. Increase to zoom in.
    img = ImageOps.crop(img, border=10)

    # Grayscale + contrast boost for thermal printing
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    # Slight extra contrast (optional)
    img = ImageEnhance.Contrast(img).enhance(1.2)

    # Resize to full width
    wpercent = target_w / float(img.size[0])
    hsize = int(float(img.size[1]) * wpercent)
    img = img.resize((target_w, hsize), Image.LANCZOS)

    # Convert to 1-bit with dithering
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

    # Scale QR to ~55% of paper width
    qr_w = int(target_w * 0.55)
    img = img.resize((qr_w, qr_w), Image.NEAREST)

    # Center on paper by padding
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

    p = Usb(VENDOR_ID, PRODUCT_ID)

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
    p.image(photo)

    # Spacing
    p.text("\n")

    # QR (optional)
    if QR_URL:
        p.text("SCAN / DOWNLOAD\n")
        qr_img = make_qr(QR_URL, PRINTER_WIDTH)
        p.image(qr_img)
        p.text("\n")

    # Footer
    p.set(align="center", bold=True)
    p.text(FOOTER_1 + "\n")
    p.set(align="center", bold=False)
    p.text(FOOTER_2 + "\n")

    p.text("\n\n")
    p.cut()

if __name__ == "__main__":
    # Change filename here, or pass your photobooth output path
    print_receipt("test.jpg")