# Deploying the tajweed-check server

The package in this repo is the whole server. Nothing is provisioned
for you — the one step that needs a human is creating the VPS and
pointing a subdomain at it. Everything after that is one command.

## 1. Pick the server

**First choice: OVHcloud VPS-1, Vint Hill Virginia — $5.35/month,
month to month, no commitment.**

- 2 vCores, 4 GB RAM, 40 GB NVMe, unlimited traffic, 1 IPv4 and a
  daily backup included.
- Verified Sept–Nov 2026 from OVHcloud's own public catalog (and
  $4.54/mo if you pay 12 months ahead). Confirm the price on the
  order page before you pay — prices move.
- Why Virginia: low latency for US users, and the server is in the
  same country as the app's first users.

**Alternative: Hetzner CX23 in Falkenstein or Helsinki, Germany /
Finland — €5.99/month including IPv4, excluding VAT (~$7.10).**

- 2 vCPU, 4 GB RAM, 40 GB NVMe, 20 TB traffic. The round trip from
  the US East Coast is ~90–110 ms, which is invisible for a batch
  check (the audio upload + inference is seconds). Mali users are
  closer to Europe than to Virginia, so this is arguably the better
  location for Sory's audience.
- Check stock at signup: CX23 has sold out in EU locations more
  than once in 2026 — try Helsinki if Falkenstein is out.

**Correction — do NOT order Hetzner in Ashburn.** The cheap CX line
is not sold in the US at all. Ashburn's cheapest 4 GB box is CPX21
at $37.49/month + $0.60 IPv4 (~$38/mo). An earlier $5–7 Ashburn
figure was wrong; the US and EU catalogs are different.

**Why 4 GB is the floor:** the model is 2.42 GB and peaks at ~3.2 GB
RAM during inference. A 2 GB box (e.g. Hetzner CPX11, $20.49/mo)
cannot run it.

## 2. Order it

1. Create an account at [ovhcloud.com](https://www.ovhcloud.com/)
   (or [hetzner.com/cloud](https://www.hetzner.com/cloud) for the
   EU alternative). OVHcloud may ask for ID verification — that is
   normal for them. The first bill is prepaid.
2. Order **VPS-1**, OS **Ubuntu 24.04**, location **Vint Hill,
   Virginia** (OVH) or **Falkenstein / Helsinki** (Hetzner CX23).
3. Note the server's **public IPv4 address** from the order
   confirmation / control panel. You will need it in the next step.
   The server comes with root access; `setup.sh` uses sudo.

## 3. Point a subdomain at it

In Cloudflare (where `saheldev.group` DNS lives), add an **A
record**:

- Name: `tajweed`
- Value: the server's public IPv4 address
- Proxy: **DNS only** (grey cloud, not orange) — Caddy on the
  server handles HTTPS itself; a proxied record would hide the real
  IP and complicate the certificate.

Wait a few minutes, then check it resolves:

```
nslookup tajweed.saheldev.group
```

## 4. Install the server (one command)

SSH into the server as root, then:

```
git clone https://github.com/somilli/tajweed-server.git
cd tajweed-server
sudo bash setup.sh tajweed.saheldev.group
```

What `setup.sh` does (Ubuntu 24.04):

- installs python3.12, venv, ffmpeg, curl, ufw; creates a 2 GB
  swapfile (the model peaks at ~3.2 GB on a 4 GB box);
- copies `app/` + `requirements.txt` to `/opt/tajweed`, builds a
  venv, installs torch (CPU) + the requirements;
- downloads the Muaalem model (~2.4 GB, a few minutes) to
  `/opt/tajweed/models/muaalem-v3_2`;
- generates an API key into `/etc/tajweed/.env` (mode 600) if none
  exists — **it prints the key once at the end; save it**;
- installs the `tajweed` systemd service (uvicorn on 127.0.0.1:8000)
  and Caddy (reverse proxy with automatic HTTPS for your domain);
- opens ports 22/80/443 in the firewall and enables it.

If the domain's A record does not point at the server yet, Caddy's
HTTPS will fail but the API still runs on port 8000 — fix the DNS
and Caddy retries automatically.

## 5. Verify it

```
curl https://tajweed.saheldev.group/health
```

Expect `{"status":"ok","model":"obadx/muaalem-model-v3_2",...}`.
Then a real check (from any machine with a wav file):

```
curl -s -F file=@clip.wav -F surah=5 -F ayah=109 \
  -H "X-API-Key: <the key setup.sh printed>" \
  https://tajweed.saheldev.group/check | head -c 600
```

## 6. Wire the app to it

The app needs two values: the base URL
(`https://tajweed.saheldev.group`) and the API key from step 4.
Every `/check` request sends the key as an `X-API-Key` header.
That wiring is a separate step in the app — the server is ready
before it.

## Running it day to day

- **Cost:** ~$5.35/month (OVH VPS-1, month to month). No per-check
  cost — the model runs on the box, not on a paid API.
- **Logs:** `sudo journalctl -u tajweed -f`
- **Restart:** `sudo systemctl restart tajweed`
- **Model location:** `/opt/tajweed/models/muaalem-v3_2`
  (re-downloaded only if deleted; OVH's daily backup covers the disk)
- **Config:** `/etc/tajweed/.env` — `TAJWEED_API_KEY`, `TAJWEED_MODEL_DIR`,
  `TAJWEED_HOST`, `TAJWEED_PORT`. Edit it, then restart the service.
  To rotate the key, change it there and restart.
- **Updating the code:** `cd ~/tajweed-server && git pull`, then
  `sudo bash setup.sh tajweed.saheldev.group` again (it will not
  overwrite an existing API key) or restart the service if only
  `app/` changed: `sudo cp -r app /opt/tajweed/ && sudo systemctl restart tajweed`.
