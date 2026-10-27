# Deploying rainy

One server runs everything behind one domain:

| Path | Service |
|---|---|
| `/` | the app (static files from this repository) |
| `/api/*` | radar server: Europe (EUMETNET OPERA) and USA (NOAA) radar with our own 2 h nowcast |
| `/om/*` | self-hosted Open-Meteo (weather model data, no rate limits, data from Open-Meteo's open-data mirror) |

When the app is opened from this domain it uses `/api` and `/om` automatically (see `API_BASE` / `OM_BASE` in `index.html`).

## Requirements

- Linux server with at least **4 vCPU, 16 GB RAM, 160 GB SSD** (e.g. Hetzner CPX42). Open-Meteo needs 8-16 GB RAM and grows its local cache on disk.
- A domain whose DNS `A` record points to the server (optionally proxied through Cloudflare).

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
curl https://$DOMAIN/api/health
```

Updates: `git pull && docker compose up -d --build`.

## Privacy

- Caddy writes no access logs; the radar server does not log requests.
- Requests to data sources use a neutral User-Agent and carry no identifying information.
- The app sends no referrer (`<meta name="referrer" content="no-referrer">`).

## Data credits (must stay visible in the app)

DWD (CC BY 4.0), EUMETNET OPERA (CC BY 4.0), EUMETSAT ("Contains modified EUMETSAT Meteosat data", CC BY 4.0),
Environment and Climate Change Canada, NOAA, NASA GIBS, Bright Sky, Open-Meteo (CC BY 4.0), OpenStreetMap / Photon.
