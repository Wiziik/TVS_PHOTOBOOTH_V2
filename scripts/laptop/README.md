# Remote access from the laptop

Run `photobooth_ssh` on your laptop → WireGuard `wg-photo` comes up → SSH shell on the Pi. Works from any network (home, event, café, tethered phone) because both the Pi and the laptop dial out to the Hetzner VPS hub.

```
Laptop ──WG──▶ Hetzner VPS (hub) ◀──WG── Photobooth Pi
                10.9.0.1
Laptop = 10.9.0.3                        Pi = 10.9.0.2
```

---

## First-time install on the Arch laptop

The two files in this directory:

- `wg-photo.conf.local` — your WireGuard client config (contains the laptop's private key — gitignored, treat as secret). Copy it to `/etc/wireguard/wg-photo.conf` on the laptop.
- `photobooth-ssh.sh` — the convenience script, mirrors `gpu-jupyter.sh`.

Steps (run on the laptop, not the Pi):

```bash
# 1. Install WireGuard config
sudo cp wg-photo.conf.local /etc/wireguard/wg-photo.conf
sudo chmod 600 /etc/wireguard/wg-photo.conf

# 2. Enable the tunnel to come up at boot (optional but convenient)
sudo systemctl enable wg-quick@wg-photo
sudo systemctl start wg-quick@wg-photo
ping 10.9.0.1                 # sanity: reach the hub
ping 10.9.0.2                 # sanity: reach the Pi

# 3. Copy the helper script + add an alias
mkdir -p ~/bash_scripts
cp photobooth-ssh.sh ~/bash_scripts/photobooth-ssh.sh
chmod +x ~/bash_scripts/photobooth-ssh.sh
echo "alias photobooth_ssh='~/bash_scripts/photobooth-ssh.sh'" >> ~/.zshrc
source ~/.zshrc

# 4. One-time: push your existing laptop SSH pubkey into the Pi
ssh-copy-id tvs@10.9.0.2

# 5. Usage
photobooth_ssh                # interactive shell on the Pi
photobooth_ssh -- journalctl -u photobooth -n 50   # one-shot command
```

---

## What the script does

1. Brings up `wg-quick@wg-photo` (systemd unit) if the interface isn't already up.
2. Pings `10.9.0.1` (the Hetzner VPS hub) until the handshake succeeds, or gives up after 25 s.
3. Pings `10.9.0.2` (the Pi) to confirm the Pi is connected.
4. Opens an SSH shell to `tvs@10.9.0.2`.

The tunnel coexists with your existing `wg0` (predapoitou) — `wg-photo` is a separate interface on a separate subnet (`10.9.0.0/24` vs `10.8.0.0/24`). You can have both up simultaneously.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[1/3]` hangs or fails | Laptop can't reach Hetzner. Check: `sudo wg show wg-photo` — is there a `latest handshake` line? If not, verify your outbound UDP works and your `/etc/wireguard/wg-photo.conf` matches the template. |
| `[2/3]` Can't reach `10.9.0.1` | Hetzner firewall / VPS is down. Check: `ssh deploy@46.224.215.98 'sudo systemctl status wg-quick@wg-photo'`. |
| `[3/3]` Can't reach Pi `10.9.0.2` | Pi offline or WG service stopped on the Pi. The Pi is set to `wg-quick@wg-photo` at boot — reboot the Pi if someone already power-cycled it. |
| SSH prompts for password | Run `ssh-copy-id tvs@10.9.0.2` once from the laptop while the tunnel is up. |

## Rotating keys

If the laptop's private key is compromised:

1. On the laptop: `wg genkey > new.key; wg pubkey < new.key > new.pub`.
2. Update `/etc/wireguard/wg-photo.conf` with the new `PrivateKey`.
3. SSH to the VPS, edit `/etc/wireguard/wg-photo.conf` server side — replace the old laptop peer's `PublicKey` with the new pub.
4. `sudo systemctl restart wg-quick@wg-photo` on both VPS and laptop.
