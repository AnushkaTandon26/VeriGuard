param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Virtual environment not found. Create it with: py -m venv venv"
    exit 1
}

& $python (Join-Path $projectRoot "app.py") --host $HostAddress --port $Port
