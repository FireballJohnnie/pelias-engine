import sys
sys.path.insert(0,'fork'); sys.path.insert(0,'lib')
import gauntlet as g
for n in ('all','persona','agent','code','reasoning'):
    v=g.parse_spec(f"tier:n=40,bypass=32,cap=7800,hot=<project>/moe_track/p1_tests/results/B_hot_sets/hot_{n}_b7.5.txt")
    print('OK',v.label, v.env.get('PELIAS_HOT_SET','')[-34:])
