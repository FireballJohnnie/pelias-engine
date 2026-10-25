#!/usr/bin/env bash
# probe_B  READ-ONLY. Gathers what is needed to write the B script (hot-bank content / live hit rate). Starts no server, writes nothing.
# Run: bash <project>/moe_track/p1_tests/probes/probe_B.sh
C=<project>; M=$C/moe_track; T=$HOME/llama-pelias-tier
echo "== tracer binary"; ls -la --time-style=long-iso $M/trace/bin | cut -c1-160
echo; echo "== tracer help (10 s limit; it may only print an error)"; timeout 10 $M/trace/bin/moe_trace --help 2>&1 | head -n 50 | cut -c1-200
echo; echo "== trace shards on disk per domain (finished .done markers)"; ls $C/logs/moe_trace/*.done 2>/dev/null | sed -E 's#.*/##; s/_p[0-9]+_s[0-9]+\.done//' | sort | uniq -c
echo; echo "== trace summary (head)"; head -c 1500 $C/logs/moe_trace/trace_summary.json 2>/dev/null; echo
echo; echo "== one prompts.jsonl and one meta.jsonl line (format)"
f=$(ls $C/logs/moe_trace/*.prompts.jsonl | head -n 1); echo "$f"; head -n 1 "$f" | cut -c1-400
f=$(ls $C/logs/moe_trace/*.meta.jsonl | head -n 1); echo "$f"; head -n 1 "$f" | cut -c1-400
echo; echo "== hot set file header"; head -n 6 $C/logs/pelias_hot_set_b7.5.txt | cut -c1-200
echo; echo "== places in the tier tree where the router output is touched (grep pelias)"
grep -n -i pelias $T/src/llama-graph.cpp 2>/dev/null | head -n 40 | cut -c1-200
echo; echo "== llama-server options mentioning moe / expert / callback"
timeout 20 $T/build-wsl/bin/llama-server --help 2>&1 | grep -n -i -E "moe|expert|callback" | head -n 20 | cut -c1-180
