# Local side of a GLM run: pull results + adapter from the pod, VERIFY, then wipe the pod.
# The adapter only moves pod -> your machine.
#   .\scripts\pull_results.ps1 -HostIp <ip> -Port <port>            # pull + verify only
#   .\scripts\pull_results.ps1 -HostIp <ip> -Port <port> -Wipe      # pull + verify + wipe pod data
param([Parameter(Mandatory)][string]$HostIp, [Parameter(Mandatory)][int]$Port, [switch]$Wipe)
$ErrorActionPreference = "Stop"
$key  = "$HOME\.ssh\your_ssh_key"
$ssh  = "C:\Windows\System32\OpenSSH\ssh.exe"; $scp = "C:\Windows\System32\OpenSSH\scp.exe"
$opts = @("-i", $key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
$dest = Join-Path (Resolve-Path "$PSScriptRoot\..") "runs\glm_run_$(Get-Date -Format yyyyMMdd_HHmm)"
New-Item -ItemType Directory -Force $dest | Out-Null

& $scp @opts -P $Port -r "root@${HostIp}:/workspace/runs/glm/*" $dest
& $scp @opts -P $Port "root@${HostIp}:/workspace/logs/setup.log" $dest 2>$null
Write-Host "pulled to $dest"

$ok = $true
$sumFile = Join-Path $dest "adapter\SHA256SUMS"
if (Test-Path $sumFile) {
    foreach ($line in Get-Content $sumFile) {
        if (-not $line.Trim()) { continue }
        $want, $name = $line -split "\s+", 2
        $got = (Get-FileHash (Join-Path $dest "adapter\$name") -Algorithm SHA256).Hash.ToLower()
        $match = $got -eq $want
        Write-Host ("{0,-32} {1}" -f $name, $(if ($match) { "OK" } else { "MISMATCH" }))
        if (-not $match) { $ok = $false }
    }
} else { Write-Host "no adapter in this run (status in results.json)"; }

if ($Wipe) {
    if (-not $ok) { throw "checksum mismatch - NOT wiping the pod" }
    & $ssh @opts -p $Port "root@$HostIp" "rm -rf /workspace/glm53-fp8 /workspace/runs /workspace/.hf /workspace/recipe; df -h /workspace | tail -1"
    Write-Host "pod data wiped. Now TERMINATE the pod in RunPod (and delete the network volume if one was attached)."
}
