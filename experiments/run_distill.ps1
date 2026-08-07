# Distillation experiment: micro (15.7M teacher) -> nano (2.9M student),
# against a from-scratch nano control on an identical token budget and seed.
#
# Batch sizing is not arbitrary. On this 6 GB card a KD step at --batch-size 64
# peaks at 8.7 GB, which Windows serves out of system RAM instead of failing:
# the run "works" at 7.2k tok/s. --batch-size 32 --grad-accum-steps 2 is the
# same 16,384 tokens/step, peaks at 4.4 GB, and runs at 34.6k tok/s.
#
# Launch detached:
#   Start-Process powershell -ArgumentList '-File','experiments\run_distill.ps1' -WindowStyle Hidden

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\pico-lm.exe"
$logs = "runs\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$common = @(
    "--data", "data", "--seq-len", "256",
    "--batch-size", "32", "--grad-accum-steps", "2",
    "--max-steps", "4000", "--warmup-steps", "200",
    "--eval-interval", "500", "--eval-iters", "40",
    "--log-interval", "100", "--seed", "1337"
)

Write-Output "=== [1/2] nano from scratch (control) ==="
& $py train --preset nano --out runs\nano-scratch @common *>&1 |
    Tee-Object -FilePath "$logs\distill-scratch.log"

Write-Output "=== [2/2] nano distilled from micro ==="
& $py train --preset nano --out runs\nano-kd --teacher runs\micro\best.pt @common *>&1 |
    Tee-Object -FilePath "$logs\distill-kd.log"

"done $(Get-Date -Format o)" | Out-File -Encoding utf8 "$logs\distill.done"
Write-Output "=== distillation experiment complete ==="
