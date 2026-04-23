#!/usr/bin/env bash
# photobooth-ssh.sh
# Arch client helper mirroring the predapoitou pattern, for the TVS photobooth:
#   - Ensures WireGuard `wg-photo` is up
#   - Waits for handshake to the Hetzner VPS hub (10.9.0.1)
#   - SSHes into the photobooth Pi over the tunnel (10.9.0.2)
#
# Works from anywhere: both the Pi and this laptop dial out to the Hetzner VPS
# (UDP 51820), which brokers the connection. No port-forwarding needed.
#
# Usage:
#   ./photobooth-ssh.sh             # interactive shell
#   ./photobooth-ssh.sh -- <cmd>    # run one-shot command and exit
#
# Install on Arch laptop:
#   1. Copy wg-photo.conf.local → /etc/wireguard/wg-photo.conf
#        sudo cp wg-photo.conf.local /etc/wireguard/wg-photo.conf
#        sudo chmod 600 /etc/wireguard/wg-photo.conf
#   2. Enable the tunnel to come up at boot (optional):
#        sudo systemctl enable wg-quick@wg-photo
#   3. Copy this script + alias:
#        cp photobooth-ssh.sh ~/bash_scripts/photobooth-ssh.sh
#        chmod +x ~/bash_scripts/photobooth-ssh.sh
#        echo "alias photobooth_ssh='~/bash_scripts/photobooth-ssh.sh'" >> ~/.zshrc
#   4. One-time SSH key install (from laptop, over the tunnel once it's up):
#        ssh-copy-id tvs@10.9.0.2

set -euo pipefail

WG_IF="wg-photo"
HUB_VPN_IP="10.9.0.1"     # Hetzner VPS inside the tunnel
PI_VPN_IP="10.9.0.2"      # Photobooth Pi inside the tunnel
PI_USER="tvs"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_root_or_sudo() {
  if [[ $EUID -ne 0 ]]; then
    if have_cmd sudo; then
      SUDO="sudo"
    else
      echo "Need root or sudo installed."
      exit 1
    fi
  else
    SUDO=""
  fi
}

wg_is_up() {
  ip link show "${WG_IF}" >/dev/null 2>&1
}

wg_up() {
  if have_cmd systemctl && systemctl list-unit-files | grep -q "^wg-quick@${WG_IF}\.service"; then
    ${SUDO} systemctl start "wg-quick@${WG_IF}"
  else
    ${SUDO} wg-quick up "${WG_IF}" >/dev/null
  fi
}

wait_for_ping() {
  local host="$1" tries="${2:-25}"
  while (( tries > 0 )); do
    if ping -c 1 -W 1 "${host}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

# ---- main ----
ensure_root_or_sudo

echo "[1/3] Ensuring WireGuard (${WG_IF}) is up..."
if ! wg_is_up; then
  wg_up
fi

echo "[2/3] Waiting for tunnel handshake (ping ${HUB_VPN_IP})..."
if ! wait_for_ping "${HUB_VPN_IP}"; then
  echo "Can't reach hub ${HUB_VPN_IP}."
  echo "Debug:  ${SUDO} wg show"
  exit 1
fi

echo "[3/3] Probing photobooth Pi at ${PI_VPN_IP}..."
if ! wait_for_ping "${PI_VPN_IP}" 15; then
  echo "Can't reach Pi ${PI_VPN_IP}."
  echo "  • Is the Pi online? (it keeps a persistent connection to the hub)"
  echo "  • ssh deploy@46.224.215.98 'sudo wg show wg-photo' to inspect server side"
  exit 1
fi

echo "[ok] SSH to ${PI_USER}@${PI_VPN_IP}"
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  exec ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    "${PI_USER}@${PI_VPN_IP}" "$@"
else
  exec ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    "${PI_USER}@${PI_VPN_IP}"
fi
