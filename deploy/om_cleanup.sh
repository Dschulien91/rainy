#!/bin/sh
# Delete old chunk files of the self-hosted Open-Meteo (server-v1.79, 20.11.2026).
#
# Why: `open-meteo sync` writes the models into time chunks (<model>/<variable>/chunk_<n>.om) and never deletes the
# ones that have fallen out of --past-days. The volume grew to 30 GB of the 75 GB disk in 12 days (disk 30 % on 12.11.,
# 69 % on 20.11.), ~6.6 GB of it chunks nobody had written for over a week. The app only asks for the last hours and
# the next days, and a chunk that is not on disk is read from Open-Meteo's open-data mirror (REMOTE_DATA_DIRECTORY),
# so deleting old chunks cannot break an answer.
#
# Safe by construction: only files named chunk_<n>.om, three levels down (<model>/<variable>/), never in static/
# (terrain), never the two newest chunks of a variable (the current one and the one before it), and only when the
# file was not modified for DAYS days.
#
# Usage (on the server, as root):  sh /opt/rainy/deploy/om_cleanup.sh [--dry-run] [DAYS]     (DAYS default 7)
# Nightly (root crontab):  40 4 * * * sh /opt/rainy/deploy/om_cleanup.sh >> /var/log/om_cleanup.log 2>&1
set -eu

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
DAYS=${1:-7}
VOL=${OM_VOLUME:-deploy_open-meteo-data}
D=$(docker volume inspect -f '{{.Mountpoint}}' "$VOL")
[ -d "$D" ] || { echo "no volume $VOL"; exit 1; }

count=0
bytes=0
for dir in "$D"/*/*/; do
  case "$dir" in */static/) continue ;; esac
  # chunk numbers of this variable, oldest first, without the newest two
  for n in $(find "$dir" -maxdepth 1 -type f -name 'chunk_*.om' -printf '%f\n' | sed -n 's/^chunk_\([0-9][0-9]*\)\.om$/\1/p' | sort -n | head -n -2); do
    f="${dir}chunk_${n}.om"
    [ -n "$(find "$f" -maxdepth 0 -mtime +"$DAYS" 2>/dev/null)" ] || continue
    size=$(stat -c %s "$f")
    count=$((count + 1))
    bytes=$((bytes + size))
    if [ "$DRY" = 1 ]; then echo "would delete ${f#"$D"/} ($((size / 1048576)) MB)"; else rm -f "$f"; fi
  done
done
echo "$(date -u '+%Y-%m-%d %H:%M')Z om_cleanup: $([ "$DRY" = 1 ] && echo 'dry run, would free' || echo 'freed') $((bytes / 1048576)) MB in $count files (older than $DAYS days)"
