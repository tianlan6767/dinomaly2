$ErrorActionPreference = 'Stop'
$py = 'D:\00_code\python_env_new\python-3.11.6-embed-amd64-deps\python.exe'
$dino = 'D:\00_code\dl_demo_v3\third_party\Dinomaly2'
$data = 'E:\AOI_EVT_B_NI\02_train\TOP\01_ader\ad_fine_crop_800_5cls'
$save = 'dinomaly2_fine_crop_800_5cls_in800'
$ckptDir = Join-Path $dino "saved_results\$save"
$log = Join-Path $ckptDir 'log.txt'

Write-Host "Waiting for training to finish (model.pth)..."
$ckpt = $null
while ($true) {
    if (Test-Path (Join-Path $ckptDir 'model.pth')) {
        $ckpt = Join-Path $ckptDir 'model.pth'
        break
    }
    $proc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -match 'run_train\.py' -and $_.CommandLine -match 'ad_fine_crop_800_5cls'
        }
    if (-not $proc) {
        $partial = Get-ChildItem $ckptDir -Filter 'model_*.pth' -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.BaseName -replace 'model_','') } |
            Select-Object -Last 1
        if (-not $partial) { throw "Training stopped but no checkpoint in $ckptDir" }
        $ckpt = $partial.FullName
        Write-Host "Using latest checkpoint: $ckpt"
        break
    }
    Start-Sleep -Seconds 120
}

$env:CUDA_VISIBLE_DEVICES = '0'
$out = Join-Path $dino 'runs\fine_crop_800_5cls_eval'
& $py -u (Join-Path $dino 'test.py') `
    --data-root $data `
    --checkpoint $ckpt `
    --image_size 798 --crop_size 798 `
    --out $out
Write-Host "Done. Results: $out"
