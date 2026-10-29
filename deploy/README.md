# Deploying rainy

One server runs everything behind one domain:

| Path | Service |
|---|---|
| `/` | the app (static files from this repository, libraries in `vendor/`) |
| `/api/*`, `/tiles/*` | radar server: DWD radar + DWD nowcast (open data), precipitation type, thunderstorm warnings, lightning, rain alerts (Web Push) |
| `/om/*` | self-hosted Open-Meteo (weather model data for the 24 h / 3-day views) |

When the app is opened from this domain it uses `/api`, `/tiles` and `/om` automatically (`API_BASE` / `OM_BASE` in `index.html`).

## Edition

`EDITION=de` (default): Germany only - the first public version. `EDITION=full` adds the Europe/USA radar with our own
nowcast and the cloud map ("sunny"); it needs about 8 GB more RAM. The app shows the matching edition by itself
(`?edition=full` only for testing).

## Requirements

- Linux server, Germany edition: **4 vCPU, 8 GB RAM, 160 GB SSD** (e.g. Hetzner CPX31). The radar server needs ~0.6 GB;
  Open-Meteo uses the rest for its cache. Full edition: 16 GB RAM (CPX42).
- A domain whose DNS `A` record points to the server, proxied through Cloudflare (orange cloud).

## Steps

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sh

# 2. Code
git clone <repository-url> rainy && cd rainy/deploy
cp .env.example .env    # set DOMAIN=...

# 3. Start (first start builds the radar server image)
docker compose up -d

# 4. Check
curl https://$DOMAIN/api/health     # versions, data age, memory
curl https://$DOMAIN/api/ready      # 200 = radar data fresh, 503 = stale (use this for uptime monitoring)
```

Updates: `git pull && docker compose up -d --build`.

## Cloudflare

- SSL/TLS mode **Full (strict)** (Caddy has a real certificate).
- Cache rule: `/tiles/*` -> eligible for cache, respect origin `Cache-Control` (tile URLs contain the data run, so they
  never change: one render per tile and run for all users).
- Cache rule: `/api/rain/point*`, `/api/thunder/*`, `/api/lightning`, `/om/*` -> eligible for cache, respect origin
  (60-600 s). The app rounds places to ~1 km, so neighbours share the same cached answer.
- Never cache `/api/push/*`.

## Monitoring and backup

- Uptime monitor (any provider) on `https://$DOMAIN/api/ready`, alert after 2 failures.
- The `push-data` volume holds the alert subscriptions and the VAPID key. Without the key every subscriber would have to
  switch alerts on again. Nightly backup, e.g. in root's crontab:
  `0 3 * * * docker run --rm -v deploy_push-data:/d -v /root/backup:/b alpine sh -c 'cp /d/* /b/'`

## Never in production

`RAINY_DEV=1` enables fake weather scenarios and a dry-run alert outbox (`server/devtools.py`) for local tests only.

## Privacy

- Caddy writes no access logs; the radar server does not log requests or IP addresses.
- Requests to data sources use a neutral User-Agent and carry no identifying information.
- Place names and place search are proxied (`/api/geo/*`), so Photon and Open-Meteo never see users.
- The app sends no referrer.

## Data credits (must stay visible in the app)

DWD (open data), EUMETSAT (MTG lightning, CC BY 4.0), Open-Meteo (CC BY 4.0), OpenFreeMap / OpenMapTiles / OpenStreetMap
(ODbL), NASA Blue Marble (satellite map design). Full edition additionally: EUMETNET OPERA, NOAA, ECCC, NASA GIBS.
