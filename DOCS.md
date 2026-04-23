# TVS Photobooth — System Documentation

**Internal-only.** Do not publish. Contains IPs, user accounts, and references to credentials.

---

## 1. What it is

A standalone street-style photobooth. The guest pays €1 at a SumUp reader, picks three poses, gets a printed thermal receipt with a QR code, scans the QR on their phone, enters their email, and receives the three photos by email. Email addresses are added to a Mailchimp audience for future campaigns.

Two physical locations, one system:
- **On-site** (the booth itself): a Raspberry Pi running the capture/print/payment loop.
- **Off-site** (Hetzner VPS): the public landing page, photo storage, Mailchimp bridge, and daily cleanup.

The Pi can be powered off every night without breaking anything — QR codes printed today keep working for 30 days because everything the guest touches lives on the VPS.

---

## 2. Architecture diagram

```
 ╔══════════════════════ ON-SITE (Raspberry Pi) ═════════════════════╗
 ║                                                                   ║
 ║   Arduino Leonardo ─── USB-MIDI Note 60 ──┐                       ║
 ║   (toggle switch)                         │                       ║
 ║                                           ▼                       ║
 ║   UVC capture card ─── OpenCV ──── photobooth.py  ◀── SPACE /F/P  ║
 ║   (HDMI/AV → USB)                     (Pygame)      (dev keys)    ║
 ║                                           │                       ║
 ║   Thermal printer    ────── ESC/POS ──────┤                       ║
 ║   (USB 0525:a700)                         │                       ║
 ║                                           │  HTTP :8080/unlock    ║
 ║                                           ▲                       ║
 ║   SumUp Solo reader ── Node daemon ───────┘                       ║
 ║                     (payment-server/server.js)                    ║
 ║                                                                   ║
 ║                           │ HTTPS                                 ║
 ╚═══════════════════════════│═══════════════════════════════════════╝
                             │
                             │  POST /api/upload   (3 JPEGs + token)
                             ▼
 ╔══════════════════ OFF-SITE (Hetzner CAX11 / nbg1) ════════════════╗
 ║                                                                   ║
 ║   Caddy :443  ──────────► Node/Express :3100 (tvs-photobooth-web) ║
 ║   (auto-HTTPS)                  │                                 ║
 ║                                 ├─► /srv/tvs-photobooth/photos    ║
 ║                                 ├─► /srv/tvs-photobooth/data      ║
 ║                                 │   (SQLite: tickets/photos/emails)║
 ║                                 └─► Mailchimp Marketing API       ║
 ║                                     (audience + tag + merge tags) ║
 ║                                                                   ║
 ║   systemd timer (daily) ──► scripts/cleanup.js ──► purge >30 days ║
 ║                                                                   ║
 ╚═══════════════════════════════════════════════════════════════════╝
                             ▲
                             │ GET /t/<ticket>
                             │
                         Guest's phone (mobile data)
```

---

## 3. Hardware

| Component | Model / ID | Role |
|---|---|---|
| Compute | Raspberry Pi (Debian/RaspiOS) | Runs the booth app + payment daemon |
| Display | HDMI monitor on `:0` | Fullscreen Pygame UI |
| Camera | USB UVC capture card `0bda:…` + analog camera | Live preview + still capture |
| Printer | Epson-class thermal (USB VID `0525`, PID `a700`) | 80 mm / 576-dot receipt prints |
| Card reader | SumUp Solo (Bluetooth or LTE) | €1 payment per session |
| Shutter button | Arduino Leonardo USB-MIDI | Sends MIDI Note 60 on toggle |
| Input | USB keyboard + mouse | Dev / kiosk fallback |

The printer and capture card compete for USB isochronous bandwidth — the code closes the camera before the printer opens. Don't remove those `camera.close()` calls.

---

## 4. Software stack

### Pi (on-site)

| Layer | Tool | Notes |
|---|---|---|
| OS | Debian / RaspiOS | Anything with Python 3.11+, SDL2, libusb |
| Photobooth app | Python 3 (`photobooth.py`) | Single-file Pygame app, 1500 LOC |
| Receipt driver | Python (`tvstore_receipt.py`) | Uses `python-escpos` over libusb |
| Web uploader | Python (`web_uploader.py`) | Background thread, disk-backed retry queue |
| Payment daemon | Node 20+ (`payment-server/server.js`) | Express, SumUp Reader Checkout API |
| Display | SDL (kmsdrm → fbcon → wayland → x11 auto-detect) | |
| Start script | Bash (`run.sh`) | Supervises both processes |

Python deps in `requirements.txt`. Node deps in `payment-server/package.json`.

### VPS (off-site)

| Layer | Tool | Version |
|---|---|---|
| OS | Debian 13 (trixie), aarch64 | Hetzner CAX11 |
| Reverse proxy / TLS | Caddy 2.11 | Auto-issues Let's Encrypt certs |
| App | Node 24 LTS + Express (`web-server/server.js`) | |
| DB | SQLite (`better-sqlite3`) | Single file at `/srv/tvs-photobooth/data/tvs-photobooth.db` |
| Templates | Nunjucks | 3 HTML templates under `web-server/views/` |
| Uploads | Multer (memory storage) | Max 10 MB × 10 files per session |
| Supervision | systemd | `tvs-photobooth-web.service` + `.cleanup.timer` |
| Firewall | UFW + Hetzner Cloud firewall | 22 / 80 / 443 only |
| Other | fail2ban, unattended-upgrades | |

---

## 5. Session data flow

One guest, end-to-end:

1. Guest taps SumUp reader. Reader tells SumUp API. Node daemon polls, sees `SUCCESSFUL`, POSTs `http://127.0.0.1:8080/unlock` → Pi credits += 1.
2. "INSERT COIN" overlay disappears on the Pi. "PAIEMENT ACCEPTÉ" shows for 1.8 s.
3. Guest flips the Leonardo toggle. USB-MIDI Note 60 arrives. `_consume_credit()` → session starts.
4. `photobooth.py` generates a Crockford base32 ticket code (8 chars, e.g. `K7M2P9QX`), creates `photos/YYYY-MM-DD/<timestamp>_K7M2P9QX_01of03_none.jpg` files as it captures.
5. Loop 3× (countdown with PRÊT between shots). Camera is only stopped after the last shot (UVC/USB bus reasons).
6. `WebUploader.queue(code, paths)` copies the 3 JPEGs to `prints_queue_web/K7M2P9QX/` and notifies its worker thread, which POSTs them to `https://photos.tvstore.fr/api/upload` with `Authorization: Bearer <upload_token>`.
7. In parallel, `PrintManager.print_async` prints a strip: header → timestamp → 3 photos stacked → QR pointing to `https://photos.tvstore.fr/t/K7M2P9QX` → footer → "don't throw away" icon → cut.
8. VPS receives upload, writes files to `/srv/tvs-photobooth/photos/K7M2P9QX/{1,2,3}.jpg`, inserts a row into `tickets` (expires in 30 days) and 3 rows into `photos`.
9. Guest scans QR → Caddy → Node. `GET /t/K7M2P9QX` renders `views/ticket.html` with photo `<img>` tags and the email form.
10. Guest submits → `POST /t/K7M2P9QX/email`:
    - Row in `emails` table.
    - Mailchimp upsert on audience `a697aa242d` with merge fields `TICKET`, `PHOTO1`, `PHOTO2`, `PHOTO3` and tag `photobooth`. `status_if_new` = `subscribed` or `transactional` depending on the opt-in checkbox.
    - Mailchimp Customer Journey (triggered by the `photobooth` tag) composes and sends the delivery email, with `*|PHOTOn|*` merge tags rendered.
11. 30 days later, the `tvs-photobooth-cleanup.timer` runs `scripts/cleanup.js`, which deletes expired `tickets` rows (ON DELETE CASCADE removes `photos` and `emails` rows) and `rm -rf`s the corresponding photo directory.

---

## 6. Infrastructure

### Hetzner Cloud

| | |
|---|---|
| Account | same as Mailchimp/OVH — manage from <https://console.hetzner.cloud> |
| Project | `TVS Photobooth` |
| Server | `tvs-photobooth` — CAX11 ARM (2 vCPU / 4 GB / 40 GB NVMe) |
| Location | `nbg1` (Nuremberg DC3) |
| Image | `debian-13` (Debian trixie) |
| IPv4 | `46.224.215.98` |
| IPv6 | `2a01:4f8:c0c:4400::1` |
| Firewall | `tvs-photobooth-web` — inbound 22 / 80 / 443 / ICMP from `0.0.0.0/0` and `::/0` |
| Backups | Automated daily, 7-day retention (`+20%` of plan) |
| Monthly cost | ≈ €5.39 gross (€4.49 + backups), billed hourly |
| SSH user | `root` (bootstrap), **`deploy`** (operations) |
| SSH key | `~/.ssh/tvs_photobooth_ed25519` on the Pi, `tvs-photobooth` key in Hetzner |

### DNS — OVH

Managed at `tvstore.fr` zone.

| Name | Type | Value |
|---|---|---|
| `photos.tvstore.fr.` | A | `46.224.215.98` |
| `photos.tvstore.fr.` | AAAA | `2a01:4f8:c0c:4400::1` |

Let's Encrypt certs are issued automatically by Caddy on first request and renewed by Caddy.

### Mailchimp

| | |
|---|---|
| Account | tvstore.fr's main Mailchimp |
| Data center | `us17` (suffix of API key) |
| Audience | "Parisian Spirit" — ID `a697aa242d` |
| Tag applied to photobooth signups | `photobooth` |
| Merge fields used | `TICKET` (text), `PHOTO1/2/3` (url) |
| Delivery mechanism | Customer Journey — **must be built once in the UI** (Automations → Customer Journeys → Trigger: Tag added = `photobooth` → Action: Send email with `*|PHOTOn|*` merge tags) |

### Registrar / emails

Domain `tvstore.fr` is on OVH. Mail sending is via Mailchimp (no custom SMTP on this project).

---

## 7. Credentials & secret storage

Store nothing in git. The repo is read-only source; real values live on each host.

| Secret | Where it lives | Used by |
|---|---|---|
| Hetzner API token | `~/.claude.json` on workstation (`hetzner` MCP env) | Provisioning only |
| Mailchimp API key | `/srv/tvs-photobooth/app/.env` on VPS (`MAILCHIMP_API_KEY`) | web-server at runtime |
| `UPLOAD_TOKEN` | `/srv/tvs-photobooth/app/.env` on VPS **and** `photobooth.ini` on Pi | Auth between Pi and VPS |
| SumUp API key | `payment-server/.env` on Pi | Payment daemon |
| SumUp merchant code + reader ID | `payment-server/.env` on Pi | Payment daemon |
| SSH private key for VPS | `~/.ssh/tvs_photobooth_ed25519` on Pi / workstation | `ssh deploy@46.224.215.98` |
| GitHub PAT for `git push` | `.git/config` on Pi (currently embedded in remote URL) | Pushing to `Wiziik/TVS_PHOTOBOOTH_V2` |

**Rotation policy:** any secret that appears in a chat transcript, screenshot, or accidentally-committed file should be rotated immediately. The `UPLOAD_TOKEN` is a single shared secret — rotate on both VPS (`/srv/tvs-photobooth/app/.env`) and Pi (`photobooth.ini`), then `systemctl restart tvs-photobooth-web` on the VPS and restart the photobooth on the Pi.

---

## 8. File layout

```
TVS_PHOTOBOOTH_V2/
├── photobooth.py          # Main Pygame app
├── tvstore_receipt.py     # ESC/POS thermal print driver
├── web_uploader.py        # Background uploader (Pi → VPS)
├── Leonardo_MIDI_Note.ino # Arduino firmware for the shutter toggle
├── run.sh                 # Launches payment-server + photobooth together
├── photobooth.service     # systemd unit (auto-start at boot)
├── photobooth.ini         # Local config — GITIGNORED, has secrets
├── photobooth.ini.example # Tracked template — copy to photobooth.ini
├── receipt_text.txt       # Printed branding, QR URL label, end-image path
├── requirements.txt       # Python deps (mido, pygame, python-escpos, requests, …)
├── logo.png               # Pay-first overlay + brand marker
├── Do_not_throw_logo.png  # Printed at bottom of each receipt
├── countdown.wav / shutter.wav / tvstore.wav
├── CLAUDE.md              # AI helper context (optional)
├── DOCS.md                # This file
├── payment-server/
│   ├── server.js          # SumUp → Pi bridge
│   ├── package.json
│   └── .env.example       # Template for SUMUP_* creds
├── web-server/            # Deployed to the Hetzner VPS
│   ├── server.js          # Express app
│   ├── mailchimp.js       # Audience upsert + tag
│   ├── mailer.js          # Transactional email stub (Customer Journey does the work)
│   ├── db.js              # better-sqlite3 wrapper
│   ├── schema.sql         # tickets / photos / emails tables
│   ├── scripts/cleanup.js # Runs daily via systemd timer
│   ├── views/             # ticket.html / sent.html / error.html
│   ├── public/style.css
│   ├── .env.example
│   └── package.json
├── photos/                # Captured sessions (GITIGNORED)
├── prints_queue/          # Fallback for print failures (GITIGNORED)
├── prints_queue_web/      # Pending web uploads (GITIGNORED)
├── logs/                  # Runtime logs (GITIGNORED)
└── .venv/                 # Python virtualenv (GITIGNORED)
```

---

## 9. Configuration files — what's tweakable without code changes

| File | Section | What to change |
|---|---|---|
| `photobooth.ini` | `[app]` | Countdown seconds, overlay text, session length (`photos_per_session`), between-shot pause |
| `photobooth.ini` | `[filters]` | Cycle list (`none,bw,retro`) |
| `photobooth.ini` | `[midi]` | Note numbers, CC mappings |
| `photobooth.ini` | `[printer]` | `image_reduce_factor`, `print_brightness`, `print_contrast`, fallback `qr_url` |
| `photobooth.ini` | `[camera]` | Resolutions, Picamera2 swap, OpenCV device index |
| `photobooth.ini` | `[web]` | `upload_url`, `upload_token`, `ticket_url_template` |
| `receipt_text.txt` | — | Brand strings, QR URL label, `END_IMAGE` path |
| `payment-server/.env` | — | `AMOUNT`, `CURRENCY`, `DESCRIPTION` |
| `web-server/.env` (on VPS) | — | `PHOTO_RETENTION_DAYS`, `MAILCHIMP_TAG`, `PORT` |

---

## 10. Ops runbook

### Starting / stopping the booth

```bash
# One-shot foreground (Ctrl-C stops both processes)
./run.sh

# As a systemd service (runs at boot, restarts on crash)
sudo systemctl start  photobooth
sudo systemctl stop   photobooth
sudo systemctl status photobooth
journalctl -u photobooth -f           # live logs
```

### First-time boot auto-start install

```bash
sudo cp photobooth.service /etc/systemd/system/photobooth.service
sudo systemctl daemon-reload
sudo systemctl enable --now photobooth
# Confirm it survives a reboot:
sudo systemctl reboot
# ssh back in and:
systemctl status photobooth
```

Requires graphical auto-login. On Raspberry Pi OS: `sudo raspi-config` → **System Options** → **Boot / Auto Login** → **Desktop Autologin**.

### Reading logs

| Log | Path |
|---|---|
| Booth app (structured) | `logs/photobooth.log` (truncated on each start) |
| Booth app (stderr) | `logs/photobooth.console.log` |
| Payment daemon | `logs/payment-server.log` |
| systemd journal (booth) | `journalctl -u photobooth -f` |
| systemd journal (web on VPS) | `ssh deploy@46.224.215.98 'journalctl -u tvs-photobooth-web -f'` |

### Manual triggers (no hardware needed)

```bash
# Grant a credit (bypass the reader):
curl -X POST http://localhost:8080/unlock

# Fire a start-note (bypass the Leonardo):
curl -X POST http://localhost:8080/trigger

# Check credits:
curl http://localhost:8080/credits

# Test a thermal print directly:
.venv/bin/python tvstore_receipt.py test.jpg 0.9

# List MIDI ports:
.venv/bin/python photobooth.py --list-midi
```

### VPS operations

```bash
ssh deploy@46.224.215.98

# Restart the landing page + upload service
sudo systemctl restart tvs-photobooth-web

# Force a cleanup run (normally daily)
sudo systemctl start tvs-photobooth-cleanup.service

# Peek the database
sqlite3 /srv/tvs-photobooth/data/tvs-photobooth.db "SELECT code, photo_count, created_at FROM tickets ORDER BY created_at DESC LIMIT 10;"

# Deploy app updates (from the Pi)
tar -cz --exclude=node_modules --exclude=.env --exclude=data --exclude=photos \
  -C web-server . \
  | ssh -i ~/.ssh/tvs_photobooth_ed25519 deploy@46.224.215.98 'tar -xz -C /srv/tvs-photobooth/app'
ssh deploy@46.224.215.98 'cd /srv/tvs-photobooth/app && npm install --omit=dev && sudo systemctl restart tvs-photobooth-web'
```

### Rotating the upload token

1. Generate: `openssl rand -hex 32`.
2. On the VPS: edit `/srv/tvs-photobooth/app/.env`, update `UPLOAD_TOKEN=`, then `sudo systemctl restart tvs-photobooth-web`.
3. On the Pi: edit `photobooth.ini` section `[web]`, update `upload_token =`, then restart the photobooth service.
4. Any pending jobs in `prints_queue_web/` on the Pi will start failing with 401 until both sides match — they'll auto-recover on the next successful POST.

---

## 10b. Remote access (WireGuard + SSH, works from anywhere)

The Hetzner VPS doubles as a WireGuard hub. Pi and laptop both dial out to it (UDP 51820), so you can SSH into the Pi regardless of where either machine is physically located — no port-forwarding at the venue required.

```
Laptop ──WG──▶ Hetzner VPS (46.224.215.98, hub) ◀──WG── Photobooth Pi
                (10.9.0.1)
Laptop = 10.9.0.3                                        Pi = 10.9.0.2
```

### Server side (already done)

- `/etc/wireguard/wg-photo.conf` on the Hetzner VPS (server pubkey: `zCQSAUgMZKVavg/0vQVwVfiB/Y1XbbbDaYSVrNIUkx8=`)
- Listens on UDP 51820 (opened in both UFW and the Hetzner Cloud firewall)
- `net.ipv4.ip_forward = 1` + `iptables FORWARD ACCEPT` on the `wg-photo` interface
- `systemctl enable wg-quick@wg-photo` — comes up at boot

### Pi side (already done)

- `/etc/wireguard/wg-photo.conf` on the Pi, `PersistentKeepalive = 25`
- `systemctl enable wg-quick@wg-photo` — tunnel established at boot, survives NAT
- Pi's WG identity at `10.9.0.2`

### Laptop side (you install once)

See `scripts/laptop/README.md`. Summary:

1. Copy `scripts/laptop/wg-photo.conf.local` (gitignored, contains your laptop private key) → `/etc/wireguard/wg-photo.conf`.
2. `sudo systemctl enable --now wg-quick@wg-photo`.
3. Copy `scripts/laptop/photobooth-ssh.sh` → `~/bash_scripts/` and alias `photobooth_ssh`.
4. `ssh-copy-id tvs@10.9.0.2` once, with the tunnel up.

Then: `photobooth_ssh` from anywhere → lands on the Pi.

### Coexistence with predapoitou

`wg-photo` is a separate interface on a separate subnet (`10.9.0.0/24` vs `wg0` on `10.8.0.0/24`). Both can be up simultaneously.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Screen stuck on "INSERT COIN" even after payment | Pi can't reach the payment daemon or credit didn't register | `journalctl -u photobooth` for errors; `curl -X POST localhost:8080/unlock` to bypass |
| "Print failed: USB device not found" | Printer unplugged or kernel driver took over | Check `lsusb`; verify `/etc/modprobe.d/tvs-photobooth.conf` has `blacklist usblp` |
| "Print failed: Access denied" | udev rule missing | Verify `/etc/udev/rules.d/99-tvs-printer.rules` exists |
| Receipt prints garbage characters | Wrong ESC/POS dialect | We ship `ESC @` + `bitImageRaster`; confirm `tvstore_receipt.py` wasn't reverted |
| QR scans → "Ticket introuvable" | Upload still pending (network), or ticket expired (>30 days) | `ls prints_queue_web/` on Pi to see pending; scan again after a minute |
| QR scans → "Ticket expiré" | Past retention window | Expected — guest needs to retake |
| Email never arrives | Mailchimp Customer Journey disabled / merge tags wrong | Verify in Mailchimp UI that journey is **On** and merge tags render in preview |
| MIDI toggle does nothing | Arduino disconnected or port renamed | `.venv/bin/python photobooth.py --list-midi`; replug Leonardo |
| Payment daemon crashes with `EADDRINUSE` | Another instance already bound `:3000` | `pkill -f "node server.js"` then restart |
| `run.sh` won't stop | Orphaned background children | `pkill -f run.sh; pkill -f photobooth.py; pkill -f "node server.js"` |

---

## 12. Full rebuild on a fresh Pi

```bash
# 1. Clone
cd ~
git clone https://github.com/Wiziik/TVS_PHOTOBOOTH_V2.git
cd TVS_PHOTOBOOTH_V2

# 2. Python venv
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt

# 3. Node (payment daemon)
cd payment-server && npm install && cd ..

# 4. Configs — copy templates, fill in secrets
cp photobooth.ini.example photobooth.ini
cp payment-server/.env.example payment-server/.env
# Edit both and fill in the real tokens (see section 7 above)

# 5. USB printer rules (sudo once)
echo 'SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ENV{ID_USB_INTERFACES}=="*:0701*:*", MODE="0660", GROUP="plugdev"' \
  | sudo tee /etc/udev/rules.d/99-tvs-printer.rules
echo 'blacklist usblp' | sudo tee /etc/modprobe.d/tvs-photobooth.conf
sudo rmmod usblp 2>/dev/null || true
sudo udevadm control --reload-rules && sudo udevadm trigger

# 6. Auto-start service
sudo cp photobooth.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now photobooth

# 7. Verify
curl http://localhost:8080/credits
journalctl -u photobooth -n 20
```

---

## 13. Full rebuild of the VPS

If the Hetzner server dies and needs replacing, the recipe is captured in our chat history. Abbreviated:

1. Create new CAX11 in `nbg1` via Hetzner console (or MCP).
2. Install Debian 13, `apt install build-essential nodejs caddy fail2ban ufw sqlite3`.
3. Create `deploy` user with SSH key from `~/.ssh/tvs_photobooth_ed25519.pub`.
4. `rsync` or `tar` the `web-server/` directory into `/srv/tvs-photobooth/app/`.
5. Copy `/srv/tvs-photobooth/app/.env` from your password manager.
6. Install systemd units: `tvs-photobooth-web.service` + `tvs-photobooth-cleanup.{service,timer}`.
7. Point Caddyfile at `photos.tvstore.fr` with `reverse_proxy 127.0.0.1:3100`.
8. Update the A/AAAA records at OVH to the new IP.
9. Change `upload_url` on the Pi if the domain moves (it shouldn't).

---

## 14. Boundaries / things that would break if you change them

- The printer's VID/PID is hardcoded as a fallback but `tvstore_receipt.py` auto-scans for any USB printer-class device. Replacing the thermal printer with an ESC/POS compatible one should just work; non-ESC/POS (e.g. STAR line mode) will not.
- The ticket code alphabet (`23456789ABCDEFGHJKLMNPQRSTUVWXYZ`) is shared between Pi and VPS — if you change it, change both.
- Photo retention is set in `web-server/.env` (`PHOTO_RETENTION_DAYS=30`). Tickets printed today will link to the retention window defined at upload time, so changing this mid-deployment only affects new sessions.
- Mailchimp merge fields `TICKET`, `PHOTO1`, `PHOTO2`, `PHOTO3` must exist on the audience. The Customer Journey template references them by name — renaming in one place without the other breaks delivery silently (no errors, just empty image tags in the email).
