$file = Join-Path $PSScriptRoot '06-overrides-responsive.css'
$lines = [System.Collections.ArrayList]@(Get-Content $file)
Write-Host "Original line count: $($lines.Count)"

# Remove in reverse order to preserve indices
$lines.RemoveRange(629, 7)   # lines 630-636: mobile ::before rules
$lines.RemoveRange(539, 26)  # lines 540-565: ::before re-enable + hover
$lines.RemoveRange(185, 165) # lines 186-350: dead hero-console CSS
$lines.RemoveRange(154, 4)   # lines 155-158: ::before { display: none }

$lines | Set-Content $file -Encoding UTF8
Write-Host "New line count: $((Get-Content $file).Count)"
