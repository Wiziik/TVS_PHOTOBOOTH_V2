/**
 * Cleanup job — delete expired tickets and their photo directories.
 * Run daily via cron or systemd timer.
 */
import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { openDb } from "../db.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const DATA_DIR  = process.env.DATA_DIR  || path.join(__dirname, "..", "data");
const PHOTO_DIR = process.env.PHOTO_DIR || path.join(__dirname, "..", "photos");

const db = openDb(DATA_DIR);

const expired = db
  .prepare("SELECT code FROM tickets WHERE expires_at < datetime('now')")
  .all();

console.log(`[cleanup] ${expired.length} expired ticket(s)`);

const del = db.prepare("DELETE FROM tickets WHERE code = ?"); // CASCADE → photos + emails
for (const { code } of expired) {
  const dir = path.join(PHOTO_DIR, code);
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch (e) {
    console.error(`[cleanup] rm ${dir}: ${e.message}`);
  }
  del.run(code);
  console.log(`[cleanup] removed ${code}`);
}

db.close();
