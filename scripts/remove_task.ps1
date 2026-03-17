Stop-ScheduledTask -TaskName 'VoiceoverMatcherWatch' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'VoiceoverMatcherWatch' -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Task removed"
