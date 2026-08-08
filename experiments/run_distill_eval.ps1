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

# Sample sizes are chosen for statistical power, not speed. The teacher/student
# gaps on the ranking probes are a couple of points; at n=200 the standard error
# on a cloze accuracy near 85% is 2.5pp, so a 2.5pp difference is one sigma and
# means nothing. n=1000 puts it near 1.1pp.
foreach ($m in $models) {
    Write-Output "=== bench $($m.name) ==="
    & $py bench --ckpt $m.ckpt --data data --pairs 2000 --cloze 1000 --gen 40 `
        --json "results\$($m.name).json" *>&1 |
        Tee-Object -FilePath "runs\logs\bench-$($m.name).log"
}

"done $(Get-Date -Format o)" | Out-File -Encoding utf8 "runs\logs\distill-eval.done"
Write-Output "=== distillation eval complete ==="
