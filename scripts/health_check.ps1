Write-Output "=== SYSTEM HEALTH CHECK ==="
Write-Output ""

# RAM
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB, 1)
$usedGB = [math]::Round($totalGB - $freeGB, 1)
$usedPct = [math]::Round(($usedGB / $totalGB) * 100, 1)
Write-Output "RAM: ${usedGB}GB / ${totalGB}GB used (${usedPct}%)"
Write-Output "RAM Free: ${freeGB}GB"
Write-Output ""

# Disk
Write-Output "=== DISK ==="
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    $freeD = [math]::Round($_.FreeSpace/1GB, 1)
    $totalD = [math]::Round($_.Size/1GB, 1)
    $usedD = [math]::Round(100 - ($_.FreeSpace/$_.Size * 100), 1)
    Write-Output "$($_.DeviceID) ${freeD}GB free / ${totalD}GB total (${usedD}% used)"
}
Write-Output ""

# CPU
Write-Output "=== CPU ==="
$cpu = Get-CimInstance Win32_Processor
Write-Output "Load: $($cpu.LoadPercentage)%"
Write-Output "Cores: $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical"
Write-Output "Clock: $($cpu.CurrentClockSpeed)MHz / $($cpu.MaxClockSpeed)MHz"
Write-Output ""

# GPU
Write-Output "=== GPU ==="
try {
    $gpu = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $parts = $gpu.Split(",").Trim()
        Write-Output "Temp: $($parts[0])C"
        Write-Output "GPU Util: $($parts[1])%"
        Write-Output "VRAM Util: $($parts[2])%"
        Write-Output "VRAM: $($parts[3])MB / $($parts[4])MB"
        Write-Output "Power: $($parts[5])W"
    }
} catch {
    Write-Output "nvidia-smi not available"
}
Write-Output ""

# Battery
Write-Output "=== BATTERY ==="
try {
    $bat = Get-CimInstance Win32_Battery
    if ($bat) {
        Write-Output "Charge: $($bat.EstimatedChargeRemaining)%"
        Write-Output "Status: $($bat.BatteryStatus)"
    } else {
        Write-Output "No battery detected (desktop)"
    }
} catch {
    Write-Output "Battery info unavailable"
}
Write-Output ""

# Python runners
Write-Output "=== EXPERIMENT RUNNERS ==="
$procs = Get-Process python3.11 -ErrorAction SilentlyContinue
Write-Output "Python processes: $($procs.Count)"
foreach ($p in $procs) {
    $memMB = [math]::Round($p.WorkingSet64/1MB, 0)
    $runtime = (Get-Date) - $p.StartTime
    $hrs = [math]::Floor($runtime.TotalHours)
    $mins = $runtime.Minutes
    Write-Output "  PID $($p.Id): ${memMB}MB RAM, running ${hrs}h${mins}m"
}
