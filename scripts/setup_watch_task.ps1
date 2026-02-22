# Remove old task first
Unregister-ScheduledTask -TaskName 'VoiceoverMatcherWatch' -Confirm:$false -ErrorAction SilentlyContinue

# Create task that runs every 5 minutes
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Bypass -File D:\_Projects\voiceover-matcher-subtitle\scripts\watch_exit.ps1'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName 'VoiceoverMatcherWatch' -Action $action -Trigger $trigger -Settings $settings -Description 'Pipeline watch - runs every 5 min' -Force

Write-Host "Task created successfully"
Get-ScheduledTask -TaskName 'VoiceoverMatcherWatch' | Get-ScheduledTaskInfo
