# Quantize the *fine-tuned* model and score the task it was fine-tuned for.
#
# The corpus sweep (run_quantize.py) shows int4 barely moving minimal pairs or
# cloze. Both of those are likelihood-ranking probes: the model only has to put
# the right continuation above three wrong ones, which survives a lot of weight
# noise. Exact-match instruction following has no such slack -- every token of
# the answer must be generated correctly. That is the probe that can actually
# tell whether int4 is safe at this scale.
#
#   Start-Process powershell -ArgumentList '-NoProfile','-File','experiments\run_quantize_sft.ps1' -WindowStyle Hidden

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\pico-lm.exe"
$tmp = "runs\_qsft"
New-Item -ItemType Directory -Force -Path $tmp, "results", "runs\logs" | Out-Null
Remove-Item "runs\logs\quantize-sft.done" -ErrorAction SilentlyContinue

$configs = @(
    @{ name = "int8-g64-emb"; bits = 8; group = 64 },
    @{ name = "int4-g64-emb"; bits = 4; group = 64 },
    @{ name = "int4-g32-emb"; bits = 4; group = 32 }
)

foreach ($c in $configs) {
    Write-Output "=== $($c.name) ==="
    & $py quantize --ckpt runs\micro-sft\best.pt --out "$tmp\$($c.name).pt" `
        --bits $c.bits --group-size $c.group --quantize-embedding
    & $py bench --ckpt "$tmp\$($c.name).pt" --data data `
        --instructions data_sft\instruct_test.jsonl `
        --pairs 300 --cloze 150 --gen 8 `
        --json "results\micro-sft-$($c.name).json"
    Remove-Item "$tmp\$($c.name).pt" -Force
}

"done" | Out-File -Encoding utf8 "runs\logs\quantize-sft.done"
Write-Output "=== quantized-SFT eval complete ==="
