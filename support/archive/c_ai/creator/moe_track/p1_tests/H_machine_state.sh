#!/usr/bin/env bash
# H  Machine state: the base config 3 times in a row while nvidia-smi logs GPU clocks, power, temperature and memory every 2 s.
# Gives the run-to-run spread of one unchanged setting plus the GPU state behind it. Windows power plan, WSL limits and RAM profile cannot be set from here.
GROUP=H_machine_state
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
MON="$OUT/H_nvidia_smi_$(date +%Y%m%d_%H%M%S).csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > "$MON" 2>&1 &
MPID=$!
trap 'kill $MPID 2>/dev/null' EXIT
run_variant base_run1 "$SET75" "$(base_flags)" "" ""
run_variant base_run2 "$SET75" "$(base_flags)" "" ""
run_variant base_run3 "$SET75" "$(base_flags)" "" ""
kill $MPID 2>/dev/null
echo; echo "--- GPU log $MON  (rows with utilisation above 50%)"
awk -F', ' '$6+0>50{n++; if(min==""||$2+0<min)min=$2+0; if($4+0>pw)pw=$4+0; if($5+0>t)t=$5+0} END{print "samples "n"   lowest SM clock "min" MHz   max power "pw" W   max temp "t" C"}' "$MON"
finish
