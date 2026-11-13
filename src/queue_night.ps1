# Night training queue (L2, RTX 4060): one job at a time, back to back. Screen-safe:
#  - each job runs at BelowNormal priority with 12 of 16 CPU threads, so the desktop stays responsive
#  - waits for >= 3 GB free RAM before starting a job
#  - skips runs that already have runs/<run_id>.json
# Each job writes oof/<run>.parquet, preds/<run>.parquet, runs/<run>.json. Log: work/queue_night.log
$ErrorActionPreference = 'Continue'
Set-Location 'E:\MLGW_Ship_It_Friday'
$py = (Resolve-Path '.venv\Scripts\python.exe').Path
$log = 'work\queue_night.log'
$env:OMP_NUM_THREADS = '12'
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }
function Run($label, $runId, [string[]]$cmd) {
    if (Test-Path "runs\$runId.json") { Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) SKIP  $label (already done)"; return }
    while ((FreeGB) -lt 3) { Start-Sleep -Seconds 20 }
    $t = Get-Date
    Add-Content $log "$($t.ToString('HH:mm:ss')) START $label (free RAM $(FreeGB) GB)"
    $out = "work\queue_last.out"
    $p = Start-Process -FilePath $py -ArgumentList $cmd -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError "work\queue_last.err"
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    Select-String -Path $out, "work\queue_last.err" -Pattern 'all-months|Traceback|Error:' | ForEach-Object { Add-Content $log "    $($_.Line)" }
    Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) END   $label (exit $($p.ExitCode), $([int]((Get-Date) - $t).TotalSeconds) s)"
}
foreach ($s in 3..8) {
    Run "xgb beh123-lh2 s$s" "L2_xgb_fw-clean_b1-f1-beh123-lh2_s$s" @('-m','src.train_xgb','--config','configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_s0.yaml','--run-id',"L2_xgb_fw-clean_b1-f1-beh123-lh2_s$s",'--seed',"$s")
    Run "xgb beh123-lh2 tc s$s" "L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s$s" @('-m','src.train_xgb','--config','configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_tc_s0.yaml','--run-id',"L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s$s",'--seed',"$s")
}
foreach ($s in 0..2) {
    Run "xgb fw1 beh123-lh2 s$s" "L2_xgb_fw1-clean_b1-f1-beh123-lh2_s$s" @('-m','src.train_xgb','--config','configs/L2_xgb_fw1-clean_b1-f1-beh123-lh2.yaml','--run-id',"L2_xgb_fw1-clean_b1-f1-beh123-lh2_s$s",'--seed',"$s")
}
foreach ($s in 0..3) {
    Run "lgbm beh123-lh2 s$s" "L2_lgbm_fw-clean_b1-f1-beh123-lh2_q_s$s" @('-m','src.train','--config',"configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_q_s$s.yaml")
}
Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) QUEUE DONE"
