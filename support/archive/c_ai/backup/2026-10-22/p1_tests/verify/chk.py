import sys,re,os
sys.path.insert(0,'fork'); sys.path.insert(0,'lib')
import gauntlet as g
H='<project>/logs'
sets={'SET75':f"tier:n=40,bypass=32,cap=7800,hot={H}/pelias_hot_set_b7.5.txt"}
specs=[sets['SET75'],sets['SET75']+',ctxcp=2',sets['SET75']+',ctxcp=8',sets['SET75']+',reuse=1',sets['SET75']+',mtp=2',sets['SET75']+',mtp=3']
for n in (40,39,38): specs.append(f"tier:n={n},bypass=32,cap=7300,hot={H}/pelias_hot_set_b7.0.txt")
for s in specs:
    try:
        v=g.parse_spec(s); print('OK ',v.label,'| flags',' '.join(v.flags)[:120],'| reuse',v.reuse,'mtp',v.mtp)
    except Exception as e: print('ERR',s,e)
