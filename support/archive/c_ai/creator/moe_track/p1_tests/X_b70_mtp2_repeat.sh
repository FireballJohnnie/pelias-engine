#!/usr/bin/env bash
# X  repeat of the 7.0 GiB + MTP draft 2 cell only (author OK 2026-10-22 21:59, after W). Question: does the b70_mtp2 12k MEAN clear 55 tok/s?
# W gave b70_mtp2 12k = 55.4 / 55.8 / 51.0 (mean 54.1), peak 11.2 GiB. This adds 6 more launches of the SAME cell (same flags as W), so the pooled n is 9.
# Same settings as W: $SET70,mtp=2, base_flags, --tests mtp. Nothing else changes.
# Run:  bash <project>/moe_track/p1_tests/X_b70_mtp2_repeat.sh      (server on port 8012 must be stopped; nothing else on the GPU)
# About 3 min per launch [ESTIMATE from W timestamps], so about 18 min. Ctrl+C is safe; the TSV so far is kept.
GROUP=X_b70_mtp2_repeat
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
for R in 1 2 3 4 5 6; do
  run_variant b70_mtp2_x$R "$SET70,mtp=2" "$(base_flags)" "" "--tests mtp"
done
finish
echo; echo "--- this series: mean over launches (FAILED left out; min-max in brackets)"
awk -F'\t' 'NR>1 && $2!="FAILED"{n++; for(i=2;i<=6;i++){s[i]+=$i; if(n==1||$i<lo[i])lo[i]=$i; if(n==1||$i>hi[i])hi[i]=$i}; if($4>=55)p++}
  END{if(n==0){print "all launches failed"; exit}
      printf "launches %d  short %.1f [%.1f-%.1f]  4k %.1f [%.1f-%.1f]  12k %.1f [%.1f-%.1f]  prompt %.0f [%.0f-%.0f]  peak max %.1f GiB  12k>=55 in %d of %d\n",n,s[2]/n,lo[2],hi[2],s[3]/n,lo[3],hi[3],s[4]/n,lo[4],hi[4],s[5]/n,lo[5],hi[5],hi[6],p+0,n}' "$TSV"
echo "W (earlier, same cell): 12k 55.4 / 55.8 / 51.0.  Pool the 12k column of both TSVs for n=9."
