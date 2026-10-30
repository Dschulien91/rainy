# Screenshot of our app (Germany, light theme) for the visual comparison with WetterOnline.
# Usage: powershell -File tools\compare_shot.ps1 [-Out file.png] [-Zoom 6] [-T 0]
param([string]$Out = "$env:TEMP\rainy_de.png", [int]$Zoom = 6, [int]$T = 0)
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$prof = Join-Path $env:TEMP "edge-compare-profile"
Remove-Item -Recurse -Force $prof -ErrorAction SilentlyContinue  # fresh profile: no cached tiles from earlier shots
Start-Process -FilePath $edge -Wait -ArgumentList @("--headless=new", "--disable-gpu", "--hide-scrollbars",
  "--user-data-dir=$prof", "--window-size=900,820", "--virtual-time-budget=25000", "--screenshot=$Out",
  "http://localhost:8770/?lat=51.2&lon=10.4&zoom=$Zoom&t=$T")
Get-Item $Out | Select-Object FullName, Length
