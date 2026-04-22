Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName = "DiscordStudyBot"
$ProjectPath = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectPath ".venv\Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectPath "requirements.txt"

Set-Location $ProjectPath

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
git pull origin main
& $PythonPath -m pip install -r $RequirementsPath
Start-ScheduledTask -TaskName $TaskName
