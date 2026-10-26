# Creates the public GitHub repository "rainy", pushes the code and switches on GitHub Pages.
# Run once: powershell -ExecutionPolicy Bypass -File C:\Users\julia\Desktop\rainy\tools\publish.ps1
# Later updates: git push (inside C:\Users\julia\Desktop\rainy)
$gh = 'C:\Program Files\GitHub CLI\gh.exe'
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
& $gh repo create rainy --public --source . --push --description 'rainy - Regenradar (Prototyp)'
& $gh api -X POST repos/Dschulien91/rainy/pages -f 'source[branch]=main' -f 'source[path]=/' | Out-Null
Write-Host ''
Write-Host 'Fertig. In 1-2 Minuten ist rainy erreichbar unter:'
Write-Host 'https://dschulien91.github.io/rainy/'
