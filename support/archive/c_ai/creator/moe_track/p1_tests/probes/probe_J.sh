#!/usr/bin/env bash
# probe_J  READ-ONLY. Gathers what is needed to write the J script (sampling and fixed output length). Starts no server, writes nothing.
# Run: bash <project>/moe_track/p1_tests/probes/probe_J.sh
SRV=$HOME/llama-pelias-tier/build-wsl/bin/llama-server
echo "== build"; timeout 20 $SRV --version 2>&1 | head -n 4 | cut -c1-200
echo; echo "== sampling and output-length options this llama-server knows"
timeout 20 $SRV --help 2>&1 | grep -n -i -E "ignore-eos|--temp|--top-k|--top-p|--min-p|--seed|repeat|presence|frequency|dry|xtc|mirostat|typical|top-n-sigma|samplers|logit-bias|backend-sampling|n-predict|reasoning|chat-template-kwargs" | cut -c1-170
