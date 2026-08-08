# Supervised fine-tuning: micro (15.7M, TinyStories) -> instruction following.
#
# The pretrained model scores 0.0% exact match on all six held-out tasks
# (results/micro-base.json), so any non-zero score here is attributable to SFT.
#
# Selection is behavioural: every eval greedy-decodes the *dev* prompts and
# best.pt tracks that exact match, never the masked loss. dev and test are
# disjoint, so the final test number is not a selection metric.
#
# Launch detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-File','experiments\run_sft.ps1' -WindowStyle Hidden

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\pico-lm.exe"
$logs = "runs\logs"
New-Item -ItemType Directory -Force -Path $logs, "results" | Out-Null
Remove-Item "$logs\sft.done" -ErrorAction SilentlyContinue

Write-Output "=== [1/2] fine-tuning micro on instruction pairs ==="
& $py finetune `
    --sft-data data_sft `
    --init-from runs\micro\best.pt `
    --out runs\micro-sft `
    --bench-jsonl data_sft\instruct_dev.jsonl `
    --select-by score `
    --batch-size 16 --max-steps 2000 --lr 3e-4 --min-lr 3e-5 `
    --warmup-steps 100 --eval-interval 250 --eval-iters 40 `
    --bench-examples 100 --log-interval 50 *>&1 |
    Tee-Object -FilePath "$logs\sft-train.log"

Write-Output "=== [2/2] behavioural eval of the fine-tuned model (held-out test) ==="
& $py bench --ckpt runs\micro-sft\best.pt --data data `
    --instructions data_sft\instruct_test.jsonl `
    --json results\micro-sft.json *>&1 |
    Tee-Object -FilePath "$logs\sft-bench.log"

"done $(Get-Date -Format o)" | Out-File -Encoding utf8 "$logs\sft.done"
Write-Output "=== SFT experiment complete ==="
