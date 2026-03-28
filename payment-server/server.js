/**
 * TVS Photobooth — SumUp payment daemon
 *
 * Creates a SumUp checkout, optionally sends it to a paired card reader,
 * polls for PAID status, triggers the printer, then immediately creates the
 * next checkout. Loops forever.
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
  status: "starting",    // starting | waiting | paid | failed | error
  api_status: null,      // raw SumUp status string
  checkout_id: null,
  checkout_reference: null,
  last_payment: null,    // { payment_id, amount, currency, timestamp }
  last_error: null,
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

async function createCheckout() {
  const reference = `PB-${Date.now()}-${randomUUID().slice(0, 8)}`;
  const body = {
    amount: AMOUNT_DECIMAL,
    currency: CURRENCY.toUpperCase(),
    checkout_reference: reference,
    description: DESCRIPTION,
    merchant_code: SUMUP_MERCHANT_CODE,
    pay_to_email: undefined,   // not needed for card reader
    // payment_type is set by the terminal itself; omitting is correct for REST checkout
  };

  const res = await fetch(`${SUMUP_BASE}/checkouts`, {
    method: "POST",
    headers: sumupHeaders(),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Create checkout HTTP ${res.status}: ${text}`);
  }

  const data = await res.json();
  if (!data.id) throw new Error(`SumUp response missing checkout id: ${JSON.stringify(data)}`);
  return data; // { id, checkout_reference, status: "PENDING", ... }
}

/**
 * Push checkout to a specific reader via SumUp Readers API.
 * Only called when SUMUP_READER_ID is set.
 * If not set, the paired reader auto-polls for PENDING checkouts associated
 * with the merchant — no explicit push needed.
 */
async function sendToReader(checkoutId) {
  if (!SUMUP_READER_ID) return;

  const res = await fetch(
    `${SUMUP_BASE}/merchants/${SUMUP_MERCHANT_CODE}/readers/${SUMUP_READER_ID}/checkout`,
    {
      method: "POST",
      headers: sumupHeaders(),
      body: JSON.stringify({ checkout_id: checkoutId }),
    }
  );

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Send-to-reader HTTP ${res.status}: ${text}`);
  }
}

async function pollCheckout(checkoutId) {
  const res = await fetch(`${SUMUP_BASE}/checkouts/${checkoutId}`, {
    headers: sumupHeaders(),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Poll checkout HTTP ${res.status}: ${text}`);
  }

  return res.json(); // { id, status: "PENDING"|"PAID"|"FAILED"|"CANCELLED", ... }
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
// Main loop
// ---------------------------------------------------------------------------

let running = true;

async function runLoop() {
  let createBackoff = 2000;

  while (running) {
    // ── Step 1: Create checkout, retry with exponential backoff ──────────────
    let checkout = null;
    while (running) {
      try {
        state.status = "starting";
        state.last_error = null;
        log("info", "Creating checkout", {
          amount: `${AMOUNT_DECIMAL} ${CURRENCY}`,
          description: DESCRIPTION,
        });

        checkout = await createCheckout();
        state.checkout_id = checkout.id;
        state.checkout_reference = checkout.checkout_reference;
        state.api_status = checkout.status; // should be "PENDING"
        createBackoff = 2000; // reset on success
        log("info", "Checkout created", {
          id: checkout.id,
          reference: checkout.checkout_reference,
        });
        break;
      } catch (err) {
        state.status = "error";
        state.last_error = err.message;
        log("error", `Checkout creation failed — retry in ${createBackoff / 1000}s`, {
          error: err.message,
        });
        await sleep(createBackoff);
        createBackoff = Math.min(createBackoff * 2, 60_000);
      }
    }

    if (!checkout) break; // only happens if running=false

    // ── Step 2: Push to reader (optional) ────────────────────────────────────
    try {
      await sendToReader(checkout.id);
      if (SUMUP_READER_ID) {
        log("info", "Checkout sent to reader", { reader_id: SUMUP_READER_ID });
      }
    } catch (err) {
      // Non-fatal: SumUp Air/Solo auto-polls for PENDING checkouts
      log("warn", "Could not push to reader (reader will auto-poll)", {
        error: err.message,
      });
    }

    // ── Step 3: Poll until terminal status ────────────────────────────────────
    state.status = "waiting";
    log("info", "Reader ready — waiting for tap...", { checkout_id: checkout.id });

    let paidData = null;
    while (running) {
      await sleep(POLL_MS);

      let data;
      try {
        data = await pollCheckout(checkout.id);
      } catch (err) {
        log("warn", "Poll error (will retry next tick)", { error: err.message });
        continue;
      }

      // Log only when the SumUp status actually changes
      if (data.status !== state.api_status) {
        log("info", "SumUp status changed", {
          from: state.api_status,
          to: data.status,
          checkout_id: checkout.id,
        });
        state.api_status = data.status;
      }

      if (data.status === "PAID") {
        paidData = data;
        state.status = "paid";
        break;
      }

      if (data.status === "FAILED" || data.status === "CANCELLED") {
        log("warn", "Checkout ended without payment — starting new cycle", {
          status: data.status,
          checkout_id: checkout.id,
        });
        state.status = "failed";
        break;
      }

      // PENDING: keep waiting
    }

    // ── Step 4: On payment, record and trigger printer ────────────────────────
    if (paidData) {
      state.cycles += 1;
      const payment = {
        payment_id: paidData.id,
        amount: AMOUNT_NUM,
        currency: CURRENCY,
        timestamp: new Date().toISOString(),
      };
      state.last_payment = payment;
      log("info", `Payment #${state.cycles} received`, payment);
      await triggerPrinter(payment.payment_id);
    }

    if (running) {
      await sleep(300); // brief pause before next checkout
      log("info", `Cycle done (total paid: ${state.cycles}) — arming reader again`);
    }
  }

  log("info", "Payment daemon stopped cleanly.");
}

// ---------------------------------------------------------------------------
// /status HTTP endpoint
// ---------------------------------------------------------------------------

const app = express();

app.get("/status", (_req, res) => {
  res.json({
    reader_active: state.status === "waiting",
    status: state.status,
    api_status: state.api_status,
    checkout_id: state.checkout_id,
    checkout_reference: state.checkout_reference,
    last_payment: state.last_payment,
    last_error: state.last_error,
    completed_cycles: state.cycles,
    uptime_seconds: Math.floor(process.uptime()),
    config: {
      amount_cents: AMOUNT_NUM,
      currency: CURRENCY,
      description: DESCRIPTION,
      reader_id: SUMUP_READER_ID ?? null,
    },
  });
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
  log("info", `Received ${signal} — stopping after current cycle`);
  running = false;
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

log("info", "TVS Photobooth payment daemon starting", {
  amount: `${AMOUNT_DECIMAL} ${CURRENCY}`,
  description: DESCRIPTION,
  merchant: SUMUP_MERCHANT_CODE,
  reader: SUMUP_READER_ID ?? "(auto-paired — reader will poll for checkouts)",
  printer: PRINTER_API_URL,
  poll_interval_ms: POLL_MS,
});

runLoop().catch((err) => {
  log("error", "Fatal error in main loop", { error: err.message, stack: err.stack });
  process.exit(1);
});
