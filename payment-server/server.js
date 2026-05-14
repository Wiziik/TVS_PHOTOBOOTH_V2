/**
 * TVS Photobooth — SumUp payment daemon (Solo terminal)
 *
 * Uses the Reader Checkout API: pushes the amount directly to the paired Solo,
 * polls the merchant transactions endpoint for the resulting client_transaction_id
 * until SUCCESSFUL, triggers the printer, then arms the reader again.
 */

import "dotenv/config";
import express from "express";
import { randomUUID } from "crypto";

// ---------------------------------------------------------------------------
// Config from env
// ---------------------------------------------------------------------------

const {
  SUMUP_API_KEY,
  SUMUP_MERCHANT_CODE,
  SUMUP_READER_ID,        // optional — find in SumUp dashboard → Devices
  PRINTER_API_URL  = "http://localhost:8080/unlock",   // photobooth.py payment gate
  AMOUNT           = "100",   // in cents; 100 = 1.00 €
  CURRENCY         = "EUR",
  DESCRIPTION      = "Photo",
  PORT             = "3000",
  POLL_INTERVAL_MS = "2000",
} = process.env;

// Validate required vars before doing anything else
const missing = [];
if (!SUMUP_API_KEY)       missing.push("SUMUP_API_KEY");
if (!SUMUP_MERCHANT_CODE) missing.push("SUMUP_MERCHANT_CODE");
if (!SUMUP_READER_ID)     missing.push("SUMUP_READER_ID");
if (missing.length) {
  console.error(`[FATAL] Missing required env vars: ${missing.join(", ")}`);
  console.error("Copy .env.example to .env and fill in the values.");
  process.exit(1);
}

const AMOUNT_NUM = parseInt(AMOUNT, 10);
if (isNaN(AMOUNT_NUM) || AMOUNT_NUM <= 0) {
  console.error("[FATAL] AMOUNT must be a positive integer in cents (e.g. 100 for 1.00 €)");
  process.exit(1);
}

// SumUp expects a decimal like 1.00, not an integer. Fix floating-point precision.
const AMOUNT_DECIMAL = parseFloat((AMOUNT_NUM / 100).toFixed(2));
const POLL_MS        = parseInt(POLL_INTERVAL_MS, 10);
const SUMUP_BASE     = "https://api.sumup.com/v0.1";

// Build auth headers fresh each request in case token is rotated via env
function sumupHeaders() {
  return {
    Authorization: `Bearer ${process.env.SUMUP_API_KEY ?? SUMUP_API_KEY}`,
    "Content-Type": "application/json",
  };
}

// ---------------------------------------------------------------------------
// State shared with /status endpoint
// ---------------------------------------------------------------------------

const state = {
  status: "idle",               // idle | starting | waiting | paid | failed | cancelled | error
  api_status: null,             // raw SumUp transaction status
  client_transaction_id: null,
  last_payment: null,           // { payment_id, amount, currency, timestamp }
  last_error: null,
  cancelled: false,
  cycles: 0,
};

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

function log(level, msg, extra = {}) {
  const ts = new Date().toISOString();
  const extras = Object.keys(extra).length ? " " + JSON.stringify(extra) : "";
  console.log(`[${ts}] [${level.toUpperCase().padEnd(5)}] ${msg}${extras}`);
}

// ---------------------------------------------------------------------------
// SumUp API helpers
// ---------------------------------------------------------------------------

/**
 * Start a checkout directly on the Solo reader.
 * Returns the client_transaction_id that we will poll for status.
 */
async function startReaderCheckout() {
  const clientTxId = `PB-${Date.now()}-${randomUUID().slice(0, 8)}`;
  const body = {
    total_amount: {
      value: AMOUNT_NUM,                 // in minor units (cents)
      currency: CURRENCY.toUpperCase(),
      minor_unit: 2,
    },
    description: DESCRIPTION,
    client_transaction_id: clientTxId,
  };

  const res = await fetch(
    `${SUMUP_BASE}/merchants/${SUMUP_MERCHANT_CODE}/readers/${SUMUP_READER_ID}/checkout`,
    {
      method: "POST",
      headers: sumupHeaders(),
      body: JSON.stringify(body),
    }
  );

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Reader checkout HTTP ${res.status}: ${text}`);
  }

  const data = await res.json().catch(() => ({}));
  // Response: { data: { client_transaction_id } } — but SumUp sometimes echoes
  // just the id we sent. Prefer server value if present, otherwise our own.
  const returned = data?.data?.client_transaction_id || clientTxId;
  return returned;
}

/**
 * Look up a transaction by client_transaction_id.
 * Returns the transaction object, or null if none exists yet (customer
 * hasn't tapped — Solo creates the transaction only at tap time).
 */
async function getTransaction(clientTxId) {
  const url = `${SUMUP_BASE}/me/transactions?client_transaction_id=${encodeURIComponent(clientTxId)}`;
  const res = await fetch(url, { headers: sumupHeaders() });

  if (res.status === 404) return null;
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Get transaction HTTP ${res.status}: ${text}`);
  }

  const data = await res.json().catch(() => null);
  // Endpoint may return an object directly or an array wrapper
  if (!data) return null;
  if (Array.isArray(data)) return data[0] || null;
  if (Array.isArray(data.items)) return data.items[0] || null;
  return data;
}

// ---------------------------------------------------------------------------
// Printer trigger (3 attempts, 1s / 2s backoff)
// ---------------------------------------------------------------------------

async function triggerPrinter(paymentId) {
  const payload = {
    payment_id: paymentId,
    amount: AMOUNT_NUM,
    currency: CURRENCY.toLowerCase(),
    timestamp: new Date().toISOString(),
  };

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch(PRINTER_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      log("info", "Printer triggered successfully", { payment_id: paymentId, attempt });
      return;
    } catch (err) {
      log("warn", `Printer attempt ${attempt}/3 failed: ${err.message}`);
      if (attempt < 3) await sleep(attempt * 1000);
    }
  }

  log("error", "Printer trigger failed after 3 attempts — continuing to next cycle", {
    payment_id: paymentId,
  });
}

// ---------------------------------------------------------------------------
// On-demand arming session
// ---------------------------------------------------------------------------
//
// The photobooth controls when the reader is armed. POST /arm starts an
// arm-cycle that keeps the Solo armed (re-arming on SumUp's ~60 s server-side
// timeout) until either a SUCCESSFUL tap → triggers /unlock → idles, or
// POST /cancel from the photobooth (timeout or user pressed the button again).

let running = true;

const session = {
  active:    false,   // true while an arm-cycle is in flight
  cancelled: false,   // flipped by /cancel; arm-cycle observes between awaits
};

async function startArmingCycle() {
  if (session.active) {
    log("warn", "/arm received while already armed — ignoring");
    return;
  }
  session.active    = true;
  session.cancelled = false;
  state.cancelled   = false;

  let armBackoff = 2000;

  try {
    while (running && !session.cancelled) {
      // ── Step 1: Push checkout to Solo reader, retry with backoff ───────────
      let clientTxId = null;
      while (running && !session.cancelled) {
        try {
          state.status = "starting";
          state.last_error = null;
          log("info", "Arming reader", {
            amount: `${AMOUNT_DECIMAL} ${CURRENCY}`,
            description: DESCRIPTION,
            reader_id: SUMUP_READER_ID,
          });

          clientTxId = await startReaderCheckout();
          state.client_transaction_id = clientTxId;
          state.api_status = "PENDING";
          armBackoff = 2000;
          log("info", "Reader armed — waiting for tap", {
            client_transaction_id: clientTxId,
          });
          break;
        } catch (err) {
          state.status = "error";
          state.last_error = err.message;
          log("error", `Reader checkout failed — retry in ${armBackoff / 1000}s`, {
            error: err.message,
          });
          await sleep(armBackoff);
          armBackoff = Math.min(armBackoff * 2, 60_000);
        }
      }

      if (!clientTxId) break;        // cancelled or shutting down

      // ── Step 2: Poll transaction by client_transaction_id ──────────────────
      state.status = "waiting";

      let paidTx = null;
      while (running && !session.cancelled) {
        await sleep(POLL_MS);
        if (session.cancelled) break;

        let tx;
        try {
          tx = await getTransaction(clientTxId);
        } catch (err) {
          log("warn", "Poll error (will retry next tick)", { error: err.message });
          continue;
        }

        if (!tx) continue;            // customer hasn't tapped yet

        if (tx.status && tx.status !== state.api_status) {
          log("info", "Transaction status changed", {
            from: state.api_status,
            to: tx.status,
            client_transaction_id: clientTxId,
          });
          state.api_status = tx.status;
        }

        if (tx.status === "SUCCESSFUL") {
          paidTx = tx;
          state.status = "paid";
          break;
        }

        if (tx.status === "FAILED" || tx.status === "CANCELLED") {
          log("warn", "Checkout ended without payment — re-arming inside same session", {
            status: tx.status,
            client_transaction_id: clientTxId,
          });
          state.status = "failed";
          break;
        }
      }

      // ── Step 3: On payment, trigger printer and end the session ────────────
      if (paidTx) {
        state.cycles += 1;
        const payment = {
          payment_id: paidTx.id || paidTx.transaction_code || clientTxId,
          amount: AMOUNT_NUM,
          currency: CURRENCY,
          timestamp: new Date().toISOString(),
        };
        state.last_payment = payment;
        log("info", `Payment #${state.cycles} received — ending arm session`, payment);
        await triggerPrinter(payment.payment_id);
        break;        // exit cycle: one paid session = one /arm
      }

      // FAILED/CANCELLED inside the session — re-arm immediately unless cancelled.
      if (session.cancelled) break;
      await sleep(300);
    }
  } finally {
    session.active    = false;
    state.status      = session.cancelled ? "cancelled" : (state.status === "paid" ? "paid" : "idle");
    state.cancelled   = session.cancelled;
    log("info", `Arm cycle ended (cancelled=${session.cancelled}, total paid=${state.cycles})`);
  }
}

function cancelArmingCycle() {
  if (!session.active) return false;
  session.cancelled = true;
  log("info", "/cancel received — current SumUp checkout will be allowed to time out on the device");
  return true;
}

// ---------------------------------------------------------------------------
// /status HTTP endpoint
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json({ limit: "32kb" }));

app.get("/status", (_req, res) => {
  res.json({
    reader_active: state.status === "waiting" || state.status === "starting",
    armed:         session.active,
    status:        state.status,
    api_status:    state.api_status,
    client_transaction_id: state.client_transaction_id,
    last_payment:  state.last_payment,
    last_error:    state.last_error,
    completed_cycles: state.cycles,
    uptime_seconds: Math.floor(process.uptime()),
    config: {
      amount_cents: AMOUNT_NUM,
      currency:     CURRENCY,
      description:  DESCRIPTION,
      reader_id:    SUMUP_READER_ID ?? null,
    },
  });
});

// Photobooth → "arm the reader now, customer just finished a session"
app.post("/arm", (_req, res) => {
  if (session.active) {
    return res.json({ ok: true, already_armed: true });
  }
  startArmingCycle().catch((err) => {
    log("error", "Arm cycle crashed", { error: err.message, stack: err.stack });
  });
  res.json({ ok: true, armed: true });
});

// Photobooth → "user pressed the button again, or 120 s window elapsed"
app.post("/cancel", (_req, res) => {
  const wasActive = cancelArmingCycle();
  res.json({ ok: true, was_active: wasActive });
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

function shutdown(signal) {
  log("info", `Received ${signal} — exiting`);
  running = false;
  process.exit(0);
}

process.on("SIGINT",  () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const port = parseInt(PORT, 10);
app.listen(port, "0.0.0.0", () => {
  log("info", `Status endpoint → http://localhost:${port}/status`);
});

log("info", "TVS Photobooth payment daemon starting (on-demand mode)", {
  amount: `${AMOUNT_DECIMAL} ${CURRENCY}`,
  description: DESCRIPTION,
  merchant: SUMUP_MERCHANT_CODE,
  reader: SUMUP_READER_ID ?? "(auto-paired)",
  printer: PRINTER_API_URL,
  poll_interval_ms: POLL_MS,
});
log("info", "Idle until POST /arm — the photobooth will request arming after each session.");
