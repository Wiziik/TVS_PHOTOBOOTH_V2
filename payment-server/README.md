# TVS Photobooth — SumUp Payment Daemon

Keeps a SumUp card reader permanently armed. Each tap triggers the printer then
immediately arms the reader again. Loops forever.

---

## Quick start — 3 steps

### 1. Get a SumUp API key

1. Go to <https://developer.sumup.com/>
2. Log in with your SumUp merchant account
3. Create an application → **Client Credentials** → generate access token
4. Copy `SUMUP_API_KEY` (starts with `sup_sk_`)
5. Your `SUMUP_MERCHANT_CODE` is visible in the SumUp dashboard (top-right profile menu)

### 2. Configure

```bash
cd payment-server
cp .env.example .env
```

Open `.env` and fill in:

```
SUMUP_API_KEY=sup_sk_your_key_here
SUMUP_MERCHANT_CODE=MYCODE
PRINTER_API_URL=http://localhost:8080/print
AMOUNT=100   # 100 cents = 1.00 €
```

Everything else has working defaults.

### 3. Run

```bash
npm install
npm start
```

That's it. The daemon logs every state change and the reader is armed immediately.

---

## Reader pairing

**SumUp Air / Solo (Bluetooth):** pair once via the SumUp app on your phone
(Settings → Card readers). After pairing, the reader auto-polls SumUp for
PENDING checkouts — no `SUMUP_READER_ID` needed.

**SumUp Solo LTE (3G/4G, standalone):** set `SUMUP_READER_ID` to your device ID
(found in SumUp dashboard → Devices). The daemon will push each checkout
directly to that reader.

---

## Pricing — two amounts

Run two instances with different `AMOUNT` values, one per reader:

```bash
# Terminal 1 — 1 € / 1 photo
AMOUNT=100 DESCRIPTION="1 Photo" PORT=3000 npm start

# Terminal 2 — 2 € / 3 photos
AMOUNT=200 DESCRIPTION="3 Photos" PORT=3001 npm start
```

Or create two `.env` files and point to them:

```bash
node -r dotenv/config server.js dotenv_config_path=.env.1photo
node -r dotenv/config server.js dotenv_config_path=.env.3photos
```

---

## Status endpoint

```
GET http://localhost:3000/status
```

```json
{
  "reader_active": true,
  "status": "waiting",
  "api_status": "PENDING",
  "checkout_id": "abc123",
  "checkout_reference": "PB-1711617600000-a1b2c3d4",
  "last_payment": {
    "payment_id": "xyz789",
    "amount": 100,
    "currency": "EUR",
    "timestamp": "2026-03-28T10:00:00.000Z"
  },
  "completed_cycles": 5,
  "uptime_seconds": 300,
  "config": {
    "amount_cents": 100,
    "currency": "EUR",
    "description": "Photo",
    "reader_id": null
  }
}
```

`status` values: `starting` → `waiting` (reader armed) → `paid` / `failed` → back to `starting`

---

## Printer payload

On each successful payment, the server POSTs to `PRINTER_API_URL`:

```json
{
  "payment_id": "abc123",
  "amount": 100,
  "currency": "eur",
  "timestamp": "2026-03-28T10:00:00.000Z"
}
```

3 retry attempts with 1s / 2s backoff. Failure is logged but does not block the
next cycle.

---

## Run as a systemd service

```ini
# /etc/systemd/system/photobooth-payment.service
[Unit]
Description=TVS Photobooth Payment Daemon
After=network.target

[Service]
WorkingDirectory=/home/pi/TVS_PHOTOBOOTH_V2/payment-server
ExecStart=/usr/bin/node server.js
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp photobooth-payment.service /etc/systemd/system/
sudo systemctl enable --now photobooth-payment
sudo journalctl -fu photobooth-payment
```
