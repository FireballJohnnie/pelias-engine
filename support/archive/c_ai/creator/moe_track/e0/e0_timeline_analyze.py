import pandas as pd, numpy as np, sys
def load(tag):
    H=[l.rstrip('\n').split('\t') for l in open(f'{tag}.host.tsv')]
    D={int(r[1]):(int(r[2]),int(r[3])) for r in H if r[0]=='D'}
    E={int(r[1]):int(r[2]) for r in H if r[0]=='E'}
    S=pd.DataFrame([[int(r[1]),int(r[2]),r[3],int(r[4]),int(r[5]),int(r[6]),int(r[7]),int(r[8])] for r in H if r[0]=='S'],
        columns='dec split be nin nn tb tl tr'.split())
    G=pd.DataFrame([[int(x) for x in l.split('\t')[1:3]]+[float(l.split('\t')[4]),int(l.split('\t')[3]),int(l.split('\t')[5])] for l in open(f'{tag}.gpu.tsv') if l[0]=='G'],columns='seq ts dur tl nn'.split())
    return D,E,S,G
def per_token(tag):
    D,E,S,G=load(tag)
    cs=S[S.be=='CUDA0'].reset_index(drop=True)
    n=min(len(cs),len(G)); cs=cs.iloc[:n].copy(); g=G.iloc[:n]
    mm=np.abs(cs.tl.values-g.tl.values)
    cs['gdur']=g.dur.values; cs['gts']=g.ts.values
    rows=[]
    for d in sorted(D):
        if d not in E or D[d][1]!=82: continue
        s=S[S.dec==d]; c=cs[cs.dec==d]
        sc=s[s.be=='CPU']
        rows.append(dict(dec=d,wall=(E[d]-D[d][0])/1e3,cpu=(sc.tr-sc.tl).sum()/1e3,gpu=c.gdur.sum(),
            hand_to_cuda=(c.tl-c.tb).sum()/1e3,hand_to_cpu=(sc.tl-sc.tb).sum()/1e3,cuda_launch=(c.tr-c.tl).sum()/1e3))
    return pd.DataFrame(rows),mm,(len(D),len(S),len(G),len(cs))
if __name__=='__main__':
    for tag in ['B_graphs_on','C_graphs_off']:
        R,mm,cnt=per_token(tag)
        print('=====',tag,'decodes/S/G/cudaS',cnt,'launch mismatch ns median/max',np.median(mm),mm.max())
        print(len(R),'single-token decodes with 82 splits')
        print(R.iloc[10:].describe().loc[['mean','50%','min','max']].round(0).drop(columns='dec').to_string())
