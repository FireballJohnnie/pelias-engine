import re, sys, hashlib, os
A1='/mnt/user-data/attachments/c7ff3710-2054-5bef-9957-5760a8e25c56/3bd97a8b-attachment.txt'
A2='/mnt/user-data/attachments/c7ff3710-2054-5bef-9957-5760a8e25c56/b41bd32d-attachment.txt'
OUT=sys.argv[1]
order=['K','A','C','D','E','F','G','H','I','L','M','N','J','B','R','U','T','P','Q','S','V']
names={'K':'K_logging','A':'A_memory_loading','C':'C_threads','D':'D_batch_sizes','E':'E_layout','F':'F_kv_attention','G':'G_scheduler_offload','H':'H_machine_state','I':'I_server_extras','L':'L_speculative_mtp','M':'M_cuda_graphs','N':'N_pinned_memory','J':'J_sampling_output','B':'B_hot_set_content','R':'R_p1_grid_load_none','U':'U_crash_sweep','T':'T_kl_gate','P':'P_bank_size_bypass9','Q':'Q_bypass32_default_load','S':'S_stock_layout_ub_load','V':'V_bypass_latency_by_length'}
sections={}   # letter -> list of (src, lineno, text)
pre=[]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def run(path,tag):
    cur=None; seenS_summary=False
    for i,l in enumerate(open(path,encoding='utf-8').read().split('\n'),1):
        if l.startswith('Shell cwd was reset'): continue
        m=re.match(r'^### ([A-Z])_\w+ / ',l)
        if m: cur=m.group(1)
        elif re.match(r'^### build hot set',l): cur='B'
        elif re.match(r'^### (tier|stock)\w*_rerun',l): cur='T'
        elif l.startswith('bypass 32, hot set'): cur='U'
        elif re.match(r'^user@Pelias:.*\$ bash ',l) or re.match(r'^bash /mnt/c/.*\.sh\s*$',l): cur=None
        elif l.startswith('no background server recorded') and seenS_summary and cur=='S': cur='V'
        if l.startswith('=== S_stock_layout_ub_load summary'): seenS_summary=True
        if cur is None: pre.append((tag,i,l))
        else: sections.setdefault(cur,[]).append((tag,i,l))
run(A1,'sweep-1'); run(A2,'sweep-2')
man=[]
for k in order:
    rows=sections.get(k,[])
    while rows and rows[-1][2].strip()=='' : rows.pop()
    fn=f"{order.index(k)+1:02d}_{names[k]}.terminal.txt"
    open(os.path.join(OUT,fn),'w',encoding='utf-8').write('\n'.join(r[2] for r in rows)+'\n')
    man.append((fn,rows[0][0],rows[0][1],rows[-1][1],len(rows)))
open(os.path.join(OUT,'00_commands_as_typed.terminal.txt'),'w',encoding='utf-8').write('\n'.join(r[2] for r in pre if r[2].strip())+'\n')
with open(os.path.join(OUT,'MANIFEST.txt'),'w') as f:
    f.write("Terminal output of the P1 test sweep, 2026-10-22 (WSL2, RTX 4070, llama.cpp tier build), cut per script from two pasted terminal sessions.\n")
    f.write("Raw text, unedited; only split by script. The 'Shell cwd was reset' tool note was dropped. Per-launch logs and result tables are separate (p1_tests/results).\n")
    f.write("O (patch isolation) terminal output was not pasted; its results are in O_patch_isolation.tsv and its logs.\n\n")
    f.write(f"source sweep-1 = K,A,C,D,E,F,G,H,I,L,M,N,J,B  sha256 {sha(A1)}  lines {len(open(A1).read().splitlines())}\n")
    f.write(f"source sweep-2 = R,U,T,P,Q,S,V                 sha256 {sha(A2)}  lines {len(open(A2).read().splitlines())}\n\n")
    f.write("file\tsource\tfirst_line\tlast_line\tlines\n")
    for r in man: f.write('\t'.join(map(str,r))+'\n')
print(open(os.path.join(OUT,'MANIFEST.txt')).read())
