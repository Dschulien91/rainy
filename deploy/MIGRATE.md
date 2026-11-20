# Moving to a new server

Checklist for moving everything to a new machine without users noticing (written 20.11.2026 for the move from the
Hetzner ARM VM to a netcup x86 VM). The images are built from the Dockerfiles, so the CPU type (ARM or x86) does not
matter. Run the commands from the operator PC; `OLD` and `NEW` are the two IPs.

## 0. Order and first login

- Image: Debian 12 or Ubuntu 24.04. When the provider lets you add an SSH key at install time, add the operator's public
  key (`~/.ssh/rainy_server.pub`) - then no password is ever needed.
- Otherwise log in once with the root password from the provider and append the public key to `/root/.ssh/authorized_keys`.

## 1. Base setup (NEW)

```bash
ssh -i ~/.ssh/rainy_server root@NEW 'set -e
apt-get update && apt-get -y upgrade && apt-get -y install git ufw sysstat
curl -fsSL https://get.docker.com | sh
# key-only login (the old server saw ~7000 password guesses a day)
printf "PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n" > /etc/ssh/sshd_config.d/00-keys-only.conf
sshd -t && systemctl reload ssh
ufw default deny incoming && ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable
# swap as a safety net, not as working memory
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo "/swapfile none swap sw 0 0" >> /etc/fstab
git clone https://github.com/Dschulien91/rainy.git /opt/rainy'
```

Test a second key login in a new terminal before closing the first one.

## 2. Data (OLD -> NEW, server to server)

`ssh -A` forwards the operator key, so OLD can write to NEW directly; nothing passes through the PC.

```bash
ssh -A -i ~/.ssh/rainy_server root@OLD 'set -e
for v in push-data verify-log eu-data eun-data na-data model-log caddy-data open-meteo-data; do
  docker run --rm -v deploy_$v:/d alpine tar cf - -C /d . |
    ssh -o StrictHostKeyChecking=accept-new root@NEW "docker volume create deploy_$v >/dev/null && docker run --rm -i -v deploy_$v:/d alpine tar xf - -C /d"
  echo "$v copied"
done'
```

- `caddy-data` holds the TLS certificates, so the new Caddy answers HTTPS before the DNS points to it.
- `open-meteo-data` (~30 GB) is faster to copy than to sync again.
- The secrets file `/opt/rainy/deploy/.env` is copied the same way, server to server, only with the owner's go for that
  move: `ssh -A root@OLD 'ssh root@NEW "cat > /opt/rainy/deploy/.env && chmod 600 /opt/rainy/deploy/.env" < /opt/rainy/deploy/.env'`.
  Never print it.

## 3. Start and check (NEW)

```bash
ssh -i ~/.ssh/rainy_server root@NEW 'cd /opt/rainy/deploy && docker compose up -d --build'
# all four instances answer through the new Caddy (DNS still on OLD):
for p in "" eu/ eun/ na/; do curl -s --resolve hellorainy.com:443:NEW https://hellorainy.com/${p}api/ready; echo; done
```

The radar instances need ~2-5 min after the start to load their last hours. Check `/api/health` for versions, the
data age and `openmeteo: true`.

## 4. Switch

1. Cloudflare -> DNS: change the `A` records of `hellorainy.com` and `www` from OLD to NEW (proxied, orange cloud).
   Users keep being served by Cloudflare; the switch takes effect within a minute.
2. Right after it, on OLD: `cd /opt/rainy/deploy && docker compose stop api api-eu api-eun api-na` - otherwise both
   servers send every rain alert and every ops push twice.
3. Copy `push-data` once more (step 2 for that volume only) and `docker compose restart api` on NEW, so alert
   subscriptions made during the move are not lost.
4. Watch `/api/ready` on all four paths and the next daily report.

## 5. Afterwards

- Keep OLD for a month as a cold standby (containers stopped, data in place). To fall back: start its containers and
  point the DNS back.
- Add the nightly backup of `push-data` (README "Monitoring and backup") on NEW, and copy it OFF the server (user
  decision 20.11.2026: buy the off-site backup space together with the netcup server, e.g. netcup storage or a
  Hetzner Storage Box; until then the 7 nightly copies in /root/backup live on the same disk).
- Measure the shared-CPU steal in the first days: `sar -u 1 60` (column `%steal`) during the 5-minute runs.
