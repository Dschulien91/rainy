# Our rain map now and at +15/+30/+60/+90 min (the same steps WetterOnline shows) for tools/fc_compare.py (files in tools/fc_shots).
# Usage: powershell -File tools\fc_capture.ps1
$dir = Join-Path $PSScriptRoot "fc_shots"
New-Item -ItemType Directory -Force $dir | Out-Null
$now = [DateTime]::UtcNow
$base = $now.AddMinutes(-($now.Minute % 5)).AddSeconds(-$now.Second)  # the app shows 5-min frames
foreach ($lead in 0, 15, 30, 60, 90) {
  $valid = $base.AddMinutes($lead).ToString("yyyyMMddHHmm")
  & (Join-Path $PSScriptRoot "compare_shot.ps1") -Out (Join-Path $dir "ours_${valid}_L$lead.png") -T $lead | Out-Null
  if ($lead -gt 0) {  # our own forecast (switch "rainy (Test)"); its L0 is the same measured image as ours_..._L0
    & (Join-Path $PSScriptRoot "compare_shot.ps1") -Out (Join-Path $dir "own_${valid}_L$lead.png") -T $lead -Fc own | Out-Null
  }
}
Get-ChildItem $dir -Filter "ours_*" | Sort-Object LastWriteTime | Select-Object -Last 6 Name
