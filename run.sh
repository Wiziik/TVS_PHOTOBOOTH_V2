#!/usr/bin/env bash
# Launches the SumUp payment server and the photobooth app together.
# Both stdout/stderr are tee'd into ./logs/. Ctrl-C stops both.

set -u
cd "$(dirname "$0")"
mkdir -p logs

PAY_LOG="logs/payment-server.log"
PB_CONSOLE_LOG="logs/photobooth.console.log"
# photobooth.py also writes logs/photobooth.log itself via logging.FileHandler

: > "$PAY_LOG"
: > "$PB_CONSOLE_LOG"

echo "[run.sh] Starting payment server -> $PAY_LOG"
( cd payment-server && node server.js ) >"$PAY_LOG" 2>&1 &
PAY_PID=$!

# Give node a moment to bind :3000 and arm the reader
sleep 2

echo "[run.sh] Starting photobooth -> logs/photobooth.log (+ $PB_CONSOLE_LOG for stderr)"
.venv/bin/python photobooth.py >"$PB_CONSOLE_LOG" 2>&1 &
PB_PID=$!

cleanup() {
  trap - INT TERM
  echo
  echo "[run.sh] Stopping (payment=$PAY_PID photobooth=$PB_PID)"
  kill "$PAY_PID" "$PB_PID" 2>/dev/null || true
  # Give them 2s to exit cleanly, then SIGKILL anything left.
  for _ in 1 2 3 4; do
    kill -0 "$PAY_PID" 2>/dev/null || kill -0 "$PB_PID" 2>/dev/null || break
    sleep 0.5
  done
  kill -9 "$PAY_PID" "$PB_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "[run.sh] Running. Payment PID=$PAY_PID  Photobooth PID=$PB_PID"
echo "[run.sh] Tail logs:  tail -f $PAY_LOG logs/photobooth.log $PB_CONSOLE_LOG"

# Exit if either child dies
while kill -0 "$PAY_PID" 2>/dev/null && kill -0 "$PB_PID" 2>/dev/null; do
  sleep 1
done

echo "[run.sh] A child process exited — stopping the other."
cleanup
