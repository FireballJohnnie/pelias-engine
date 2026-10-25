#!/usr/bin/env bash
# W  7.0 GiB bank follow-up (author's request 2026-10-22, after the C to N, J, B sweep). One command runs everything in order, nothing to type between cells.
# Why 7.0 GiB: the 7.5 GiB base peaks at 10.6 GiB; MTP adds about 0.9 GiB (10.7 -> 11.6, L) and -ub 4096 about 0.6 GiB (10.6 -> 11.2, D); the VRAM line is 11.2 GiB.
# A 7.0 GiB bank peaked at 10.2 GiB with -ub 2048 (E, one launch). Whether MTP or ub 4096 fits below 11.2 at this size is what this tests.
# 5 cells x 3 launches, ROUND-ROBIN (round 1 of every cell, then round 2, then round 3) so slow drift in the machine does not favour one cell.
# A cell that does not fit in VRAM is logged as FAILED and the run continues. Stop it with Ctrl+C; the table so far is already in the TSV.
# Run:  bash <project>/moe_track/p1_tests/W_bank70_extras.sh      (server on port 8012 must be stopped; nothing else on the GPU)
GROUP=W_bank70_extras
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
UB4="-t $T -tb $TB $LM -b 4096 -ub 4096"
for R in 1 2 3; do
  run_variant b70_base_r$R         "$SET70"        "$(base_flags)" "" ""
  run_variant b70_ub4096_r$R       "$SET70"        "$UB4"          "" ""
  run_variant b70_mtp2_r$R         "$SET70,mtp=2"  "$(base_flags)" "" "--tests mtp"
  run_variant b70_mtp3_r$R         "$SET70,mtp=3"  "$(base_flags)" "" "--tests mtp"
  run_variant b70_mtp3_ub4096_r$R  "$SET70,mtp=3"  "$UB4"          "" "--tests mtp"
done
finish
echo; echo "--- mean per cell over its launches (FAILED launches left out; min-max in brackets)"
awk -F'\t' 'NR>1 && $2!="FAILED"{c=$1; sub(/_r[0-9]+$/,"",c); n[c]++; for(i=2;i<=6;i++){s[c,i]+=$i; if(!((c,i) in lo)||$i<lo[c,i])lo[c,i]=$i; if(!((c,i) in hi)||$i>hi[c,i])hi[c,i]=$i}; if(!(c in o)){o[c]=++k; nm[k]=c}}
  NR>1 && $2=="FAILED"{c=$1; sub(/_r[0-9]+$/,"",c); f[c]++; if(!(c in o)){o[c]=++k; nm[k]=c}}
  END{printf "%-18s %-4s %-4s %-17s %-17s %-17s %-19s %s\n","cell","ok","fail","short","4k","12k","prompt@12k","peak GiB(max)";
      for(j=1;j<=k;j++){c=nm[j]; if(n[c]>0) printf "%-18s %-4d %-4d %5.1f [%4.1f-%4.1f] %5.1f [%4.1f-%4.1f] %5.1f [%4.1f-%4.1f] %7.0f [%5.0f-%5.0f] %s\n",c,n[c],f[c]+0,s[c,2]/n[c],lo[c,2],hi[c,2],s[c,3]/n[c],lo[c,3],hi[c,3],s[c,4]/n[c],lo[c,4],hi[c,4],s[c,5]/n[c],lo[c,5],hi[c,5],hi[c,6]; else printf "%-18s %-4d %-4d all launches failed\n",c,0,f[c]}}' "$TSV"
