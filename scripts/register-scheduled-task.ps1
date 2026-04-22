Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName = "DiscordStudyBot"
$ProjectPath = Split-Path -Parent $PSScriptRoot
$RunScriptPath = Join-Path $PSScriptRoot "run-bot.ps1"

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$RunScriptPath`"" `
    -WorkingDirectory $ProjectPath

$Trigger = New-ScheduledTaskTrigger -AtStartup

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Runs the Discord study RAG bot" `
    -Force

Start-ScheduledTask -TaskName $TaskName
