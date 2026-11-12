#!/usr/bin/env bash
# Sabaq tajweed-check server — one-shot setup for Ubuntu 24.04.
# Usage (on the VPS, from the tajweed-server folder):
#   sudo bash setup.sh tajweed.saheldev.group
# The domain must already point at this server's IP (A record), or
# Caddy cannot issue the HTTPS certificate and the setup will fail at
# the last step (the API itself will still run on port 8000).
set -euo pipefail

DOMAIN="${1:?usage: sudo bash setup.sh <domain>  (e.g. tajweed.saheldev.group)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
INSTALL=/opt/tajweed
ENV_FILE=/etc/tajweed/.env

echo "== packages =="
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv ffmpeg curl ufw > /dev/null

echo "== 2 GB swap (the model peaks ~3.2 GB; the box has 4 GB) =="
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile > /dev/null
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== copy app =="
mkdir -p "$INSTALL" /etc/tajweed
cp -r "$SRC/app" "$SRC/requirements.txt" "$INSTALL/"

echo "== python venv =="
python3.12 -m venv "$INSTALL/venv"
"$INSTALL/venv/bin/pip" install -q --upgrade pip
"$INSTALL/venv/bin/pip" install -q torch==2.15.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$INSTALL/venv/bin/pip" install -q -r "$INSTALL/requirements.txt"

echo "== model (2.42 GB from Hugging Face) =="
"$INSTALL/venv/bin/huggingface-cli" download obadx/muaalem-model-v3_2 \
  --local-dir "$INSTALL/models/muaalem-v3_2" > /dev/null

echo "== api key =="
if [ ! -f "$ENV_FILE" ]; then
  KEY="$(openssl rand -hex 32)"
  {
    echo "TAJWEED_API_KEY=$KEY"
    echo "TAJWEED_MODEL_DIR=$INSTALL/models/muaalem-v3_2"
    echo "TAJWEED_HOST=127.0.0.1"
    echo "TAJWEED_PORT=8000"
  } > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi

echo "== systemd service =="
cp "$SRC/tajweed.service" /etc/systemd/system/tajweed.service
systemctl daemon-reload
systemctl enable --now tajweed

echo "== caddy (reverse proxy + auto HTTPS) =="
apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https > /dev/null
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | tee /etc/apt/sources.list.d/caddy-stable.list > /dev/null
apt-get update -qq
apt-get install -y -qq caddy > /dev/null
sed "s/__DOMAIN__/$DOMAIN/" "$SRC/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

echo "== firewall =="
ufw allow 22/tcp > /dev/null
ufw allow 80/tcp > /dev/null
ufw allow 443/tcp > /dev/null
ufw --force enable > /dev/null

echo
echo "== done =="
echo "Health:  https://$DOMAIN/health"
echo "API key: $(grep TAJWEED_API_KEY "$ENV_FILE" | cut -d= -f2)"
echo "(Save the key — it is only printed here. The app sends it as the X-API-Key header.)"
sleep 5
curl -sf "https://$DOMAIN/health" || curl -sf http://127.0.0.1:8000/health
