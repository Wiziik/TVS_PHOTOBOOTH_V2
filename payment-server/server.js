/**
 * TVS Photobooth — SumUp payment daemon (Solo terminal)
 *
 * Uses the Reader Checkout API: pushes the amount directly to the paired Solo,
 * polls the merchant transactions endpoint for the resulting client_transaction_id
 * until SUCCESSFUL, then triggers the printer. Exactly one checkout per /arm —
 * the reader is never re-armed automatically, so an untapped offer leaves at
 * most one abandoned payment in the SumUp app instead of one per ~65 s timeout.
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
  // Hard backstop: stop polling a single checkout after this long if nothing
  // terminal arrives (the photobooth normally ends it sooner via /cancel). Set
  // above the Solo's ~65 s on-device timeout so a last-second tap is still seen.
  READER_CHECKOUT_TIMEOUT_MS = "75000",
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
const CHECKOUT_TTL_MS = parseInt(READER_CHECKOUT_TIMEOUT_MS, 10);
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
// The photobooth controls when the reader is armed. POST /arm pushes ONE
// checkout to the Solo (fired right after the customer takes a photo) and waits
// for that single checkout to resolve: a SUCCESSFUL tap → triggers /unlock →
// idles, or it ends (decline / device timeout / CHECKOUT_TTL_MS backstop /
// POST /cancel from the photobooth) with no payment. It is never re-armed
// automatically — one button-press = one checkout = at most one abandoned
// payment, instead of a fresh checkout every ~65 s while the offer is up.

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

  try {
    // ── Step 1: Push exactly ONE checkout to the Solo reader ─────────────────
    // Bounded retry only covers a transient API/network hiccup while *pushing*
    // the checkout — on failure nothing reaches the device, so a retry can't
    // create a duplicate. After this point we never push another checkout.
    let clientTxId = null;
    let armBackoff = 2000;
    for (let attempt = 1; attempt <= 3 && running && !session.cancelled; attempt++) {
      try {
        state.status = "starting";
        state.last_error = null;
        log("info", "Arming reader (single checkout)", {
          amount: `${AMOUNT_DECIMAL} ${CURRENCY}`,
          description: DESCRIPTION,
          reader_id: SUMUP_READER_ID,
        });

        clientTxId = await startReaderCheckout();
        state.client_transaction_id = clientTxId;
        state.api_status = "PENDING";
        log("info", "Reader armed — waiting for tap", {
          client_transaction_id: clientTxId,
        });
        break;
      } catch (err) {
        state.status = "error";
        state.last_error = err.message;
        log("error", `Reader checkout failed (attempt ${attempt}/3)`, { error: err.message });
        if (attempt < 3) await sleep(armBackoff);
        armBackoff = Math.min(armBackoff * 2, 10_000);
      }
    }

    if (!clientTxId) return;        // cancelled, shutting down, or push failed

    // ── Step 2: Poll this single checkout to its conclusion ──────────────────
    // Ends on a SUCCESSFUL tap, a FAILED/CANCELLED result, the device timeout
    // (CHECKOUT_TTL_MS backstop), or /cancel — then the session is over. There
    // is no outer loop, so a timed-out offer is never re-armed.
    state.status = "waiting";
    const armedAt   = Date.now();
    let paidTx      = null;
    let txSeen      = false;   // a transaction exists (customer has tapped)
    let cancelGrace = 2;       // after /cancel, poll a couple more times so a tap
                               // that crossed the cancel boundary still lands

    while (running) {
      if (Date.now() - armedAt > CHECKOUT_TTL_MS) {
        log("info", "Checkout window elapsed without payment — session over (no re-arm)", {
          client_transaction_id: clientTxId,
        });
        state.status = "failed";
        break;
      }
      if (session.cancelled) {
        if (!txSeen || cancelGrace <= 0) break;
        cancelGrace -= 1;
      }

      await sleep(POLL_MS);

      let tx;
      try {
        tx = await getTransaction(clientTxId);
      } catch (err) {
        log("warn", "Poll error (will retry next tick)", { error: err.message });
        continue;
      }

      if (!tx) continue;            // customer hasn't tapped yet
      txSeen = true;

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
        log("info", "Checkout ended without payment — session over (no re-arm)", {
          status: tx.status,
          client_transaction_id: clientTxId,
        });
        state.status = "failed";
        break;
      }
    }

    // ── Step 3: On a successful tap, trigger the printer ─────────────────────
    if (paidTx) {
      state.cycles += 1;
      const payment = {
        payment_id: paidTx.id || paidTx.transaction_code || clientTxId,
        amount: AMOUNT_NUM,
        currency: CURRENCY,
        timestamp: new Date().toISOString(),
      };
      state.last_payment = payment;
      log("info", `Payment #${state.cycles} received`, payment);
      await triggerPrinter(payment.payment_id);
    }
  } finally {
    session.active    = false;
    state.status      = session.cancelled ? "cancelled" : (state.status === "paid" ? "paid" : "idle");
    state.cancelled   = session.cancelled;
    log("info", `Arm session ended (cancelled=${session.cancelled}, total paid=${state.cycles})`);
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

// Photobooth → "user pressed the button again, or the print-offer window elapsed"
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
  checkout_ttl_ms: CHECKOUT_TTL_MS,
});
log("info", "Idle until POST /arm — the photobooth will request arming after each session.");
