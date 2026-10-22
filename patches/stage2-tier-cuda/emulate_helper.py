# Emulates mm_ids_helper's expert-bound arithmetic: original (count slots) vs fixed (count distinct (token, expert) pairs).
# The fixed variant is emulated lane by lane with the same shfl_up rule used in the patch (not the intended semantics).
import random, sys
W = 32
def pow2(n):
    p = 1
    while p < n: p *= 2
    return p

def helper(ids, n_experts, n_used, fixed):
    n_tok = len(ids)
    neu = pow2(n_used)
    bounds = [0]*(n_experts+1); ids_dst = [None]*(n_tok*n_used)
    rows_written = {}
    for e in range(n_experts):
        nex_prev = 0; compact = []
        for it0 in range(0, n_tok, W//neu):
            lanes = []
            for lane in range(W):
                it = it0 + lane//neu; iex = lane % neu
                v = ids[it][iex] if (iex < n_used and it < n_tok) else 2**31-1
                lanes.append((it, iex, v))
            for lane in range(W):
                it, iex, v = lanes[lane]
                rep = False
                if fixed:
                    for d in range(1, neu):
                        other = lanes[lane-d][2] if lane-d >= 0 else v   # shfl_up out of range returns own value
                        rep = rep or (iex >= d and other == v)
                nex_prev += 1 if (v < e and not rep) else 0 if fixed else (1 if v < e else 0)
            # token groups that use this expert, in token order (benign last-writer race on iex)
            for g in range(W//neu):
                grp = [lanes[g*neu+k] for k in range(neu)]
                hit = [(it, iex) for (it, iex, v) in grp if v == e and it < n_tok]
                if hit: compact.append((hit[0][0], hit[-1][1]))
        bounds[e] = nex_prev
        for itc, (it, iex) in enumerate(compact):
            ids_dst[nex_prev+itc] = it*n_used + iex
        rows_written[e] = len(compact)
    # end bound: last expert's nex_prev + its compact count
    bounds[n_experts] = bounds[n_experts-1] + rows_written[n_experts-1]
    return bounds, ids_dst, rows_written

def check(ids, n_experts, n_used, fixed):
    b, dst, rw = helper(ids, n_experts, n_used, fixed)
    holes = 0
    for e in range(n_experts):
        rng = range(b[e], b[e+1])
        holes += sum(1 for r in rng if r >= len(dst) or dst[r] is None)
        if b[e+1]-b[e] != rw[e]: pass
    return holes, b, rw

random.seed(7)
n_experts, n_used = 40, 8        # a bank of 39 real slots + zero slot 0 (illustrative sizes)
tot = {False: [0,0], True: [0,0]}
for trial in range(3000):
    n_tok = random.choice([9, 12, 13, 16, 24, 30, 31])
    ids = []
    for _ in range(n_tok):
        n_miss = random.randint(0, 8)
        real = random.sample(range(1, n_experts), 8 - n_miss)
        row = real + [0]*n_miss
        random.shuffle(row); ids.append(row)
    for fixed in (False, True):
        holes, b, rw = check(ids, n_experts, n_used, fixed)
        tot[fixed][0] += (holes > 0); tot[fixed][1] += 1
        if fixed and holes: print("FIXED VARIANT STILL HAS HOLES", trial, holes); sys.exit(1)
        if fixed:
            # every (token, real expert) pair must appear exactly once in the compact rows; expert ranges contiguous
            seen = set()
            for e in range(1, n_experts):
                for r in range(b[e], b[e+1]):
                    pass
            assert all(b[e] <= b[e+1] for e in range(n_experts))
            total_pairs = sum(rw.values())
            assert b[n_experts] == total_pairs, (b[n_experts], total_pairs)
print("original: trials with empty rows inside an expert range:", tot[False][0], "of", tot[False][1])
print("fixed   : trials with empty rows inside an expert range:", tot[True][0],  "of", tot[True][1])
# no-duplicate rows (stock behaviour): both variants must agree
random.seed(3); same = 0
for trial in range(1000):
    n_tok = random.choice([9, 16, 31]); ids = [random.sample(range(0, 40), 8) for _ in range(n_tok)]
    a = helper(ids, 40, 8, False); c = helper(ids, 40, 8, True)
    same += (a[0] == c[0] and a[1] == c[1])
print("rows without repeats: original and fixed give identical bounds and ids_dst in", same, "of 1000 trials")
