# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A two-process Raspberry Pi photobooth: `photobooth.py` (Python/pygame, fullscreen UI + camera + thermal printer + MIDI) and `payment-server/server.js` (Node, SumUp card-reader daemon). They communicate over localhost HTTP — the payment daemon POSTs `/unlock` to grant a credit, the UI spends one credit per capture. An Arduino Leonardo running `Leonardo_MIDI_Note.ino` is a USB-MIDI toggle switch that sends Note 60 to trigger the shutter.

## Run

```
./run.sh                                      # payment server + photobooth together, Ctrl-C stops both
.venv/bin/python photobooth.py                # photobooth alone
cd payment-server && node server.js           # payment server alone
.venv/bin/python photobooth.py --list-midi    # list MIDI input ports and exit
.venv/bin/python tvstore_receipt.py [PATH] [REDUCE] [QR_URL]   # print a receipt directly (test.jpg if no arg)
```

Logs land in `./logs/` (`payment-server.log`, `photobooth.log`, `photobooth.console.log`). `photobooth.service` is a systemd unit for installed deployments; note it hard-codes `/home/pi/TVS_PHOTOBOOTH_V2/venv/…` while local dev uses `.venv/…` — the two paths diverge on purpose.

## Architecture

### One photo, end to end

1. **Arduino Leonardo** toggle → USB-MIDI Note 60 (middle C). Equivalent: `POST /trigger` on port 8080.
2. `MidiController` thread in `photobooth.py` normalises events into `self.midi_queue`; the main loop drains it in `handle_midi_event`.
3. Start-note only fires if `_consume_credit()` returns true — otherwise the UI shows "Please pay first".
4. Countdown → `CameraController.capture_still_and_stop()` → save JPEG + `.json` sidecar under `photos/YYYY-MM-DD/` → queue for print.
5. `PrintManager` runs `tvstore_receipt.print_receipt()` on a background thread so the UI never freezes. On failure it copies the photo to `prints_queue/` as a fallback.

### Payment path

`payment-server/server.js` pushes a checkout to a SumUp Solo reader (Reader Checkout API), polls `/me/transactions?client_transaction_id=…` until `SUCCESSFUL`, then POSTs `PRINTER_API_URL` (default `http://localhost:8080/unlock`). The photobooth's embedded `HTTPServer` (`TRIGGER_PORT = 8080`) increments a thread-safe credit counter. Endpoints: `POST /unlock` (add credit), `POST /trigger` (inject start-note MIDI event), `GET /credits` (read count).

### Modules

- `photobooth.py` — single-file app. `AppConfig` (dataclass) is built from `photobooth.ini`; MIDI, camera, print manager, unlock HTTP server, pygame UI are all wired up in `PhotoboothApp.__init__`.
- `tvstore_receipt.py` — ESC/POS thermal printer driver. VID/PID hardcoded `0x04B8:0x0E15` (Epson-class, 80 mm / 576 dots) — the stale docstring in `photobooth.py` header says otherwise; trust the code. Uses a custom `_ChunkedUsb` backend that writes in 4 KiB chunks with a 100 ms delay to avoid buffer overflow, and `_unbind_printer_from_kernel()` that unbinds only the printer's `usblp`/`cdc_acm` interfaces via sysfs — it will not touch the Arduino.
- `Leonardo_MIDI_Note.ino` — Arduino sketch, debounced toggle on pin 2, uses the `MIDIUSB` library. Sends Note On/Off on state change only.

### Camera backends

`CameraController` tries Picamera2/libcamera first, then auto-scans OpenCV V4L2 indices 0–4, then falls back to a `mock` backend that synthesises frames so the app runs headlessly. `capture_still_and_stop()` grabs a frame from the already-running preview stream and fully closes the camera before printing: UVC capture cards return a black frame on the first read after a mode switch, and leaving the camera's isochronous USB transfers active competes with the printer's bulk transfers on a shared bus.

## Configuration

- `photobooth.ini` — auto-created with defaults on first run. Sections: `[app] [filters] [midi] [printer] [camera]`. Defaults: MIDI note 60 = start, 38 = cycle filter, 40 = print last, 41 = delete (two-press confirm), CC 1 = brightness, CC 2 = countdown length.
- `receipt_text.txt` — printed branding, footer, QR URL. Auto-created; keys must be uppercase (`KEY = value`). Empty `QR_URL` disables the QR block.
- `payment-server/.env` — SumUp creds (`SUMUP_API_KEY`, `SUMUP_MERCHANT_CODE`, `SUMUP_READER_ID`, `AMOUNT` in cents, `PRINTER_API_URL`). Template in `.env.example`.

## Dev / debug shortcuts

Keyboard inside the pygame window (no MIDI needed): `SPACE` = capture (still requires a credit), `F` = cycle filter, `P` = print last, `ESC`/`Q` = quit. Without a SumUp reader: `curl -X POST localhost:8080/unlock` to grant a credit, `curl -X POST localhost:8080/trigger` to simulate the start-note MIDI event. SDL driver is auto-detected (`kmsdrm` → `fbcon` → `wayland` → `x11`); set `SDL_VIDEODRIVER` to pin it.

## Gotchas

- **USB bus contention**: both `_print_last()` and `_start_photo_sequence()` call `camera.close()` and sleep ~1 s before printing to free isochronous bandwidth. Don't remove — a UVC capture card and a thermal printer on the same hub will stall otherwise.
- **Credits are in-memory only**: `PhotoboothApp._payment_credits` is a plain list, reset on restart. Persist it if that matters.
- **Stale log at repo root**: `./photobooth.log` predates the move to `logs/`; `logs/photobooth.log` is the live file.
