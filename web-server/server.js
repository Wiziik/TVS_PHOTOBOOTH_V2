import "dotenv/config";
import express from "express";
import multer from "multer";
import nunjucks from "nunjucks";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { openDb } from "./db.js";
import { upsertSubscriber } from "./mailchimp.js";
import { sendPhotoDeliveryEmail } from "./mailer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────────────
const PORT                 = parseInt(process.env.PORT || "3100", 10);
const BASE_URL             = (process.env.BASE_URL || "http://localhost:3100").replace(/\/$/, "");
const DATA_DIR             = process.env.DATA_DIR  || path.join(__dirname, "data");
const PHOTO_DIR            = process.env.PHOTO_DIR || path.join(__dirname, "photos");
const PHOTO_RETENTION_DAYS = parseInt(process.env.PHOTO_RETENTION_DAYS || "30", 10);
const UPLOAD_TOKEN         = process.env.UPLOAD_TOKEN || "";
const MAILCHIMP_API_KEY    = process.env.MAILCHIMP_API_KEY || "";
const MAILCHIMP_LIST_ID    = process.env.MAILCHIMP_LIST_ID || "";
const MAILCHIMP_TAG        = process.env.MAILCHIMP_TAG || "photobooth";

if (!UPLOAD_TOKEN) {
  console.error("FATAL: UPLOAD_TOKEN not set");
  process.exit(1);
}

fs.mkdirSync(PHOTO_DIR, { recursive: true });
const db = openDb(DATA_DIR);

// ── Templating ────────────────────────────────────────────────────────────
nunjucks.configure(path.join(__dirname, "views"), {
  autoescape: true,
  noCache: process.env.NODE_ENV !== "production",
});
function render(res, tpl, ctx) {
  res.type("html").send(nunjucks.render(tpl, ctx));
}

// ── Ticket codes ──────────────────────────────────────────────────────────
// Crockford base32 minus 0/O/I/1 to avoid OCR confusion on the thermal print.
const ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
function randomTicketCode(len = 8) {
  const bytes = crypto.randomBytes(len);
  return Array.from(bytes, (b) => ALPHABET[b % ALPHABET.length]).join("");
}

// ── Helpers ───────────────────────────────────────────────────────────────
function isValidCode(code) {
  return typeof code === "string" && /^[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{6,16}$/.test(code);
}

function bearer(req) {
  const h = req.headers.authorization || "";
  return h.startsWith("Bearer ") ? h.slice(7) : "";
}

function photoUrl(code, seq) {
  return `${BASE_URL}/p/${code}/${seq}.jpg`;
}

// ── Upload (from the Pi) ──────────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024, files: 10 }, // 10 MB × up to 10 files
});

const app = express();
app.disable("x-powered-by");
app.set("trust proxy", 1); // behind Caddy

app.use(express.urlencoded({ extended: false, limit: "16kb" }));
app.use(express.json({ limit: "16kb" }));
app.use("/public", express.static(path.join(__dirname, "public"), { maxAge: "7d" }));

app.get("/healthz", (_req, res) => res.json({ ok: true }));

app.post("/api/upload", upload.array("photos", 10), (req, res) => {
  if (bearer(req) !== UPLOAD_TOKEN) {
    return res.status(401).json({ error: "unauthorized" });
  }
  let { code } = req.body;
  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: "no photos" });
  if (files.length > 10) return res.status(400).json({ error: "too many photos" });

  if (code) {
    if (!isValidCode(code)) return res.status(400).json({ error: "bad code" });
  } else {
    code = randomTicketCode();
  }

  // Ensure uniqueness in case of collision on caller-supplied code.
  const exists = db.prepare("SELECT 1 FROM tickets WHERE code = ?").get(code);
  if (exists) return res.status(409).json({ error: "code already exists" });

  const ticketDir = path.join(PHOTO_DIR, code);
  fs.mkdirSync(ticketDir, { recursive: true });

  const expiresAt = new Date(Date.now() + PHOTO_RETENTION_DAYS * 86400_000).toISOString();

  const tx = db.transaction(() => {
    db.prepare(
      "INSERT INTO tickets (code, expires_at, photo_count) VALUES (?, ?, ?)",
    ).run(code, expiresAt, files.length);

    const insertPhoto = db.prepare(
      "INSERT INTO photos (ticket_code, seq, filename, size_bytes) VALUES (?, ?, ?, ?)",
    );
    files.forEach((f, i) => {
      const seq = i + 1;
      const filename = `${seq}.jpg`;
      fs.writeFileSync(path.join(ticketDir, filename), f.buffer);
      insertPhoto.run(code, seq, filename, f.size);
    });
  });
  tx();

  console.log(`[upload] ticket=${code} photos=${files.length}`);
  res.json({
    ok: true,
    code,
    url: `${BASE_URL}/t/${code}`,
    expires_at: expiresAt,
  });
});

// ── Landing page ──────────────────────────────────────────────────────────
function loadTicket(code) {
  if (!isValidCode(code)) return null;
  const ticket = db.prepare("SELECT * FROM tickets WHERE code = ?").get(code);
  if (!ticket) return null;
  if (new Date(ticket.expires_at) < new Date()) return { expired: true, ...ticket };
  return ticket;
}

app.get("/t/:code", (req, res) => {
  const ticket = loadTicket(req.params.code);
  if (!ticket) {
    return res.status(404).type("html").send(
      nunjucks.render("error.html", {
        title: "Ticket introuvable",
        message: "Ce ticket n'existe pas ou a été supprimé.",
      }),
    );
  }
  if (ticket.expired) {
    return res.status(410).type("html").send(
      nunjucks.render("error.html", {
        title: "Ticket expiré",
        message: `Les photos sont conservées ${PHOTO_RETENTION_DAYS} jours et ont été supprimées.`,
      }),
    );
  }

  const photos = db
    .prepare("SELECT seq FROM photos WHERE ticket_code = ? ORDER BY seq")
    .all(req.params.code);

  const alreadyEmailed = db
    .prepare("SELECT 1 FROM emails WHERE ticket_code = ?")
    .get(req.params.code);

  render(res, "ticket.html", {
    ticket,
    photoUrls: photos.map((p) => photoUrl(ticket.code, p.seq)),
    alreadyEmailed: !!alreadyEmailed,
    retentionDays: PHOTO_RETENTION_DAYS,
  });
});

app.post("/t/:code/email", async (req, res) => {
  const ticket = loadTicket(req.params.code);
  if (!ticket || ticket.expired) {
    return res.status(404).type("html").send(
      nunjucks.render("error.html", {
        title: "Ticket introuvable",
        message: "Ce ticket n'existe pas ou a expiré.",
      }),
    );
  }

  const email = String(req.body.email || "").trim().toLowerCase();
  const optIn = req.body.marketing_opt_in === "1" || req.body.marketing_opt_in === "on";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).type("html").send(
      nunjucks.render("error.html", {
        title: "Adresse invalide",
        message: "Vérifie ton adresse e-mail puis recommence.",
      }),
    );
  }

  const photoRows = db
    .prepare("SELECT seq FROM photos WHERE ticket_code = ? ORDER BY seq")
    .all(ticket.code);
  const photoUrls = photoRows.map((p) => photoUrl(ticket.code, p.seq));

  const result = db
    .prepare(
      "INSERT INTO emails (ticket_code, email, marketing_opt_in) VALUES (?, ?, ?)",
    )
    .run(ticket.code, email, optIn ? 1 : 0);
  const emailRowId = result.lastInsertRowid;

  // Mailchimp: upsert subscriber + tag. Non-fatal on failure.
  if (MAILCHIMP_API_KEY && MAILCHIMP_LIST_ID) {
    try {
      await upsertSubscriber({
        apiKey: MAILCHIMP_API_KEY,
        listId: MAILCHIMP_LIST_ID,
        email,
        marketingOptIn: optIn,
        tag: MAILCHIMP_TAG,
        mergeFields: {
          TICKET: ticket.code,
          PHOTO1: photoUrls[0] || "",
          PHOTO2: photoUrls[1] || "",
          PHOTO3: photoUrls[2] || "",
        },
      });
      db.prepare("UPDATE emails SET mailchimp_status='added' WHERE id=?").run(emailRowId);
    } catch (e) {
      console.error("[mailchimp] failed:", e.message);
      db.prepare(
        "UPDATE emails SET mailchimp_status='error', mailchimp_error=? WHERE id=?",
      ).run(String(e.message).slice(0, 500), emailRowId);
    }
  } else {
    db.prepare("UPDATE emails SET mailchimp_status='skipped' WHERE id=?").run(emailRowId);
  }

  // Transactional delivery (stubbed — plug in SMTP / Brevo / Mandrill later)
  try {
    await sendPhotoDeliveryEmail({
      to: email,
      ticketCode: ticket.code,
      photoUrls,
      baseUrl: BASE_URL,
    });
  } catch (e) {
    console.error("[mailer] failed:", e.message);
  }

  render(res, "sent.html", { ticket, email });
});

// ── Photo download ────────────────────────────────────────────────────────
app.get("/p/:code/:n.jpg", (req, res) => {
  const { code, n } = req.params;
  if (!isValidCode(code)) return res.status(404).end();
  const seq = parseInt(n, 10);
  if (!Number.isFinite(seq) || seq < 1 || seq > 50) return res.status(404).end();

  const ticket = db.prepare("SELECT expires_at FROM tickets WHERE code = ?").get(code);
  if (!ticket || new Date(ticket.expires_at) < new Date()) {
    return res.status(410).end();
  }
  const file = path.join(PHOTO_DIR, code, `${seq}.jpg`);
  if (!fs.existsSync(file)) return res.status(404).end();
  res.setHeader("Cache-Control", "private, max-age=3600");
  res.sendFile(file);
});

// ── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, "127.0.0.1", () => {
  console.log(
    `tvs-photobooth-web on :${PORT} | BASE_URL=${BASE_URL} | ` +
      `data=${DATA_DIR} photos=${PHOTO_DIR} retention=${PHOTO_RETENTION_DAYS}d | ` +
      `mailchimp=${MAILCHIMP_API_KEY ? "on" : "off"}`,
  );
});

process.on("SIGTERM", () => process.exit(0));
process.on("SIGINT",  () => process.exit(0));
