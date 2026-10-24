#!/bin/bash
# P1B single-invocation launcher (Claude R85). Usage: p1b_launch.sh <frozen binding> <expected binding sha256>
# Reads GPU index, output_dir and timeout from the binding; judges admission on that GPU only (no compute process and
# >= 26624 MiB free); exclusive mkdir of a fresh launch dir and output dir; ONE producer call under
# timeout -k <grace> <wall>; records UTC/uptime/exit code; never retries, resumes or changes GPU. Touches no foreign process.
set -u
B=$1; WANT=$2
REPO=/home/dongyicheng/OpenWAM; PY=/home/dongyicheng/miniconda3/envs/openwam/bin/python
cd "$REPO" || exit 1
[ "$(sha256sum "$B" | cut -d' ' -f1)" = "$WANT" ] || { echo "binding sha mismatch" >&2; exit 2; }
read -r O G W K COMMIT < <($PY -c "import json,sys;b=json.load(open(sys.argv[1]));print(b['output_dir'],b['gpu_index'],b['budget']['wall_s'],b['budget']['kill_grace_s'],b['repo_commit'])" "$B")
L=${O}_launch
mkdir "$L" || { echo "launch dir exists: $L" >&2; exit 3; }
pre() { echo "$1" > "$L/not_started"; exit 4; }
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || pre HEAD_MISMATCH
[ -z "$(git status --porcelain --untracked-files=no)" ] || pre DIRTY
U=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader) || pre NO_GPU
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free --format=csv,noheader > "$L/admission_gpus.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$L/admission_compute_apps.txt"
FREE=$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits)
BUSY=$(grep -c "$U" "$L/admission_compute_apps.txt")
echo "gpu=$G uuid=$U free_MiB=$FREE compute_apps_on_target=$BUSY threshold_MiB=26624" > "$L/admission_decision.txt"
{ [ "$BUSY" = 0 ] && [ "$FREE" -ge 26624 ]; } || pre "ADMISSION_REFUSED free=$FREE busy=$BUSY"
echo admitted >> "$L/admission_decision.txt"
mkdir "$O" || pre OUTDIR_EXISTS
( while true; do echo "$(date -u +%FT%T.%3NZ) $(nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$L/gpu_mem_samples.txt"; sleep 5; done ) &
SAMP=$!
date -u +%FT%T.%3NZ > "$L/start_utc"; cat /proc/uptime > "$L/start_uptime"
CUDA_VISIBLE_DEVICES=$G timeout -k "$K" "$W" $PY -I scripts/probe/p1b_claude.py run --binding "$B" --out "$O" > "$O/console.log" 2>&1
RC=$?
date -u +%FT%T.%3NZ > "$L/end_utc"; cat /proc/uptime > "$L/end_uptime"; echo $RC > "$L/exit_code"
kill $SAMP
exit $RC
