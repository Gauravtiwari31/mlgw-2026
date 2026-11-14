# Day training queue (L2, RTX 4060), 14 Nov: more members for the blend, one job at a time, screen-safe
# (BelowNormal priority, 12 CPU threads, waits for 3 GB free RAM, skips finished runs). Log: work/queue_day.log
$ErrorActionPreference = 'Continue'
Set-Location 'E:\MLGW_Ship_It_Friday'
$py = (Resolve-Path '.venv\Scripts\python.exe').Path
$log = 'work\queue_day.log'
$env:OMP_NUM_THREADS = '12'
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }
function Run($label, $runId, [string[]]$cmd) {
    if (Test-Path "runs\$runId.json") { Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) SKIP  $label (already done)"; return }
    while ((FreeGB) -lt 3) { Start-Sleep -Seconds 20 }
    $t = Get-Date
    Add-Content $log "$($t.ToString('HH:mm:ss')) START $label (free RAM $(FreeGB) GB)"
    $p = Start-Process -FilePath $py -ArgumentList $cmd -NoNewWindow -PassThru -RedirectStandardOutput "work\queue_day_last.out" -RedirectStandardError "work\queue_day_last.err"
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    Select-String -Path "work\queue_day_last.out", "work\queue_day_last.err" -Pattern 'all-months|Traceback|Error:' | ForEach-Object { Add-Content $log "    $($_.Line)" }
    Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) END   $label (exit $($p.ExitCode), $([int]((Get-Date) - $t).TotalSeconds) s)"
}
foreach ($s in 0..3) {
    Run "xgb fw1 tc s$s" "L2_xgb_fw1-clean_b1-f1-beh123-lh2_tc_s$s" @('-m','src.train_xgb','--config','configs/L2_xgb_fw1-clean_b1-f1-beh123-lh2_tc.yaml','--run-id',"L2_xgb_fw1-clean_b1-f1-beh123-lh2_tc_s$s",'--seed',"$s")
}
foreach ($s in 0..1) {
    Run "catboost tc s$s" "L2_catboost_fw-clean_b1-f1-beh123-lh2_tc_s$s" @('-m','src.train_catboost','--config',"configs/L2_catboost_fw-clean_b1-f1-beh123-lh2_tc_s$s.yaml")
}
foreach ($s in 9..14) {
    Run "xgb tc s$s" "L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s$s" @('-m','src.train_xgb','--config','configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_tc_s0.yaml','--run-id',"L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s$s",'--seed',"$s")
}
Add-Content $log "$((Get-Date).ToString('HH:mm:ss')) QUEUE DONE"
