Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectPath = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectPath ".venv\Scripts\python.exe"
$MainPath = Join-Path $ProjectPath "main.py"

Set-Location $ProjectPath
& $PythonPath $MainPath
