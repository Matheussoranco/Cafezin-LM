# Score the distillation experiment behaviourally: teacher vs distilled student
# vs the same-size student trained from scratch on an identical token budget.
#
# Training loss cannot be compared across these three at all — the KD student's
# loss is a blend of cross-entropy and KL against the teacher, so its number is
# on a different scale than the control's. The behavioural probes are the only
# thing that compares.
#
#   Start-Process powershell -ArgumentList '-NoProfile','-File','experiments\run_distill_eval.ps1' -WindowStyle Hidden

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\pico-lm.exe"
New-Item -ItemType Directory -Force -Path "results", "runs\logs" | Out-Null
Remove-Item "runs\logs\distill-eval.done" -ErrorAction SilentlyContinue

$models = @(
    @{ name = "micro-teacher";  ckpt = "runs\micro\best.pt" },
    @{ name = "nano-kd";        ckpt = "runs\nano-kd\best.pt" },
    @{ name = "nano-scratch";   ckpt = "runs\nano-scratch\best.pt" }
)

foreach ($m in $models) {
    Write-Output "=== bench $($m.name) ==="
    & $py bench --ckpt $m.ckpt --data data --pairs 400 --cloze 200 --gen 16 `
        --json "results\$($m.name).json" *>&1 |
        Tee-Object -FilePath "runs\logs\bench-$($m.name).log"
}

"done $(Get-Date -Format o)" | Out-File -Encoding utf8 "runs\logs\distill-eval.done"
Write-Output "=== distillation eval complete ==="
