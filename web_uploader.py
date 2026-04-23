"""Background uploader: ships photo sessions to the TVS web server.

Jobs are persisted to disk under prints_queue_web/<ticket_code>/ so a crash or
reboot while the Pi is offline doesn't lose them — on next startup the queued
jobs are picked up and retried. Single worker thread, exponential backoff.
"""
from __future__ import annotations

import logging
import random
import shutil
import threading
from pathlib import Path
from typing import Iterable

import requests


TICKET_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def generate_ticket_code(length: int = 8) -> str:
    """Crockford-ish base32 without 0/O/1/I/L — unambiguous on thermal print."""
    rng = random.SystemRandom()
    return "".join(rng.choice(TICKET_ALPHABET) for _ in range(length))


class WebUploader:
    def __init__(
        self,
        upload_url: str,
        upload_token: str,
        queue_dir: Path,
        *,
        timeout: float = 60.0,
        initial_retry: float = 10.0,
        max_retry: float = 600.0,
    ) -> None:
        self.upload_url = upload_url.strip()
        self.upload_token = upload_token.strip()
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.initial_retry = initial_retry
        self.max_retry = max_retry

        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._pending: list[str] = []
        self._load_pending()

        self._thread = threading.Thread(target=self._worker, name="WebUploader", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return bool(self.upload_url and self.upload_token)

    def _load_pending(self) -> None:
        if not self.queue_dir.exists():
            return
        for d in sorted(self.queue_dir.iterdir()):
            if d.is_dir() and any(d.glob("*.jpg")):
                self._pending.append(d.name)
        if self._pending:
            logging.info("WebUploader: %d pending job(s) on disk", len(self._pending))

    def queue(self, ticket_code: str, photo_paths: Iterable[Path]) -> None:
        if not self.enabled:
            logging.warning("WebUploader disabled (missing url/token); not queuing %s", ticket_code)
            return
        job_dir = self.queue_dir / ticket_code
        job_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(photo_paths, start=1):
            shutil.copy2(str(p), job_dir / f"{i}.jpg")
        with self._cv:
            self._pending.append(ticket_code)
            self._cv.notify()
        logging.info("WebUploader queued: %s (%d photos)", ticket_code, i)

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def _worker(self) -> None:
        backoff = self.initial_retry
        while not self._stop.is_set():
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    return
                ticket = self._pending[0]
            try:
                self._upload(ticket)
                shutil.rmtree(self.queue_dir / ticket, ignore_errors=True)
                with self._cv:
                    if self._pending and self._pending[0] == ticket:
                        self._pending.pop(0)
                backoff = self.initial_retry
                logging.info("WebUploader: %s uploaded OK", ticket)
            except Exception as e:
                logging.warning(
                    "WebUploader: %s failed: %s (retry in %.0fs)", ticket, e, backoff
                )
                with self._cv:
                    self._cv.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self.max_retry)

    def _upload(self, ticket_code: str) -> None:
        job_dir = self.queue_dir / ticket_code
        photos = sorted(job_dir.glob("*.jpg"))
        if not photos:
            return
        open_files = [(p, open(p, "rb")) for p in photos]
        try:
            files = [("photos", (p.name, fh, "image/jpeg")) for p, fh in open_files]
            r = requests.post(
                self.upload_url,
                headers={"Authorization": f"Bearer {self.upload_token}"},
                data={"code": ticket_code},
                files=files,
                timeout=self.timeout,
            )
        finally:
            for _, fh in open_files:
                try:
                    fh.close()
                except Exception:
                    pass
        if r.status_code == 409:
            logging.info("WebUploader: %s already on server (409) — treating as success", ticket_code)
            return
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
