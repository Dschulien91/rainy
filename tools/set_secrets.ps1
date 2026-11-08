# Puts the login secrets into deploy/.env on the production server and restarts the api.
# Run it yourself in PowerShell (it asks for each value; secrets are typed hidden, never shown or stored locally, and
# go to the server only through the encrypted SSH connection's input, not as command-line arguments):
#   powershell -ExecutionPolicy Bypass -File C:\Users\julia\Desktop\rainy\tools\set_secrets.ps1
# Leave a value empty to keep what is already on the server.
$server = "root@2.31.201.83"
$key = Join-Path $env:USERPROFILE ".ssh\rainy_server"

function Ask($name, $label, [switch]$Secret) {
  if ($Secret) {
    $s = Read-Host "$label (verdeckt)" -AsSecureString
    $v = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))
  } else { $v = Read-Host $label }
  if ($v) { return "$name=$($v.Trim())" } else { return $null }
}

$lines = @(
  (Ask "GOOGLE_CLIENT_ID" "Google OAuth Client-ID (endet auf .apps.googleusercontent.com)"),
  (Ask "GOOGLE_CLIENT_SECRET" "Google OAuth Client-Secret" -Secret),
  (Ask "SMTP_HOST" "SMTP-Server (All-Inkl, z. B. w017ff82.kasserver.com)"),
  (Ask "SMTP_USER" "SMTP-Benutzer (bei All-Inkl der Postfach-Login, z. B. hey@hellorainy.com)"),
  (Ask "SMTP_PASS" "SMTP-Passwort des Postfachs" -Secret)
) | Where-Object { $_ }

$names = ($lines | ForEach-Object { ($_ -split "=", 2)[0] }) -join "|"
$script = @"
set -e
cd /opt/rainy/deploy
touch .env && chmod 600 .env
if [ -n '$names' ]; then grep -vE '^($names)=' .env > .env.new || true; mv .env.new .env; fi
cat >> .env <<'RAINY_SECRETS_END'
$($lines -join "`n")
RAINY_SECRETS_END
grep -q '^AUTH_SECRET=' .env || echo "AUTH_SECRET=`$(openssl rand -hex 32)" >> .env
grep -q '^MAIL_FROM=' .env || echo 'MAIL_FROM=rainy <hey@hellorainy.com>' >> .env
grep -q '^SMTP_PORT=' .env || echo 'SMTP_PORT=465' >> .env
sed -i 's/\r$//' .env
chmod 600 .env
docker compose up -d api >/dev/null 2>&1
echo "gespeichert, api neu gestartet"
"@
$script.Replace("`r`n", "`n") | ssh -i $key -o BatchMode=yes $server "bash -s"
