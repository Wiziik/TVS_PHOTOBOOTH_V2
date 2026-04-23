CREATE TABLE IF NOT EXISTS tickets (
  code         TEXT PRIMARY KEY,
  created_at   DATETIME NOT NULL DEFAULT (datetime('now')),
  expires_at   DATETIME NOT NULL,
  photo_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS photos (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_code  TEXT NOT NULL,
  seq          INTEGER NOT NULL,
  filename     TEXT NOT NULL,
  size_bytes   INTEGER NOT NULL,
  FOREIGN KEY (ticket_code) REFERENCES tickets(code) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS emails (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_code       TEXT NOT NULL,
  email             TEXT NOT NULL,
  marketing_opt_in  INTEGER NOT NULL DEFAULT 0,
  created_at        DATETIME NOT NULL DEFAULT (datetime('now')),
  mailchimp_status  TEXT,
  mailchimp_error   TEXT,
  FOREIGN KEY (ticket_code) REFERENCES tickets(code) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_photos_ticket  ON photos(ticket_code);
CREATE INDEX IF NOT EXISTS idx_emails_ticket  ON emails(ticket_code);
CREATE INDEX IF NOT EXISTS idx_tickets_expire ON tickets(expires_at);
