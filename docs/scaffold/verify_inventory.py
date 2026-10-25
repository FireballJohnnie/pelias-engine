#!/usr/bin/env python3
"""Verify the pelias-engine tree against its two manifests. Standard library only.

    python3 docs/scaffold/verify_inventory.py            (run from anywhere; the root is found from this file)

Checks
  1. every row of docs/scaffold/REPO_INVENTORY.tsv: the file exists and its SHA-256 equals the recorded one;
  2. every file of the tree outside support/archive and support/local_only is listed there (two files are
     left out on purpose: the inventory itself and docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md, which records the result);
  3. every pushable row of support/MANIFEST.tsv (PUSH-VERBATIM, PUSH-REDACTED, TIER-BUILD): the copy exists and
     its SHA-256 equals repo_sha256; every DUPLICATE-OF-REPO row: the named repository file exists and has the same hash;
  4. every file under support/archive is listed in the manifest;
  5. support/local_only/ is ignored by git and no file under it is tracked (when git is available);
  6. if support/local_only/MANIFEST_LOCAL.tsv is present (the author's working copy, not a clone): its files match.

Exit status 0 when every check passes, 1 otherwise. Source hashes in the manifest identify the originals on the
author's drive; they cannot be rechecked from a clone and are not checked here.
"""
import csv, hashlib, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
INV = 'docs/scaffold/REPO_INVENTORY.tsv'
MAN = 'support/MANIFEST.tsv'
LOCAL_MAN = 'support/local_only/MANIFEST_LOCAL.tsv'
SCAFFOLD = 'docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md'
problems = []
notes = []


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def p(msg):
    problems.append(msg)
    print('PROBLEM', msg)


# 1 and 2: inventory
inv = rows(INV)
listed = set()
n_ok = 0
for r in inv:
    rel = r['path']
    listed.add(rel)
    full = os.path.join(ROOT, rel)
    if not os.path.isfile(full):
        p('missing: ' + rel)
    elif sha(full) != r['sha256']:
        p('changed: ' + rel)
    else:
        n_ok += 1
print('inventory: %d rows, %d match' % (len(inv), n_ok))

skip_dirs = ('.git', 'support/archive', 'support/local_only')
tree = set()
for dp, dn, fn in os.walk(ROOT):
    rel_dir = os.path.relpath(dp, ROOT).replace(os.sep, '/')
    rel_dir = '' if rel_dir == '.' else rel_dir
    dn[:] = [d for d in dn if not any(((rel_dir + '/' + d) if rel_dir else d) == s for s in skip_dirs)]
    for f in fn:
        tree.add((rel_dir + '/' + f) if rel_dir else f)
left_out = {INV, SCAFFOLD}
for rel in sorted(tree - listed - left_out):
    p('not in inventory: ' + rel)
print('tree: %d files outside support/archive and support/local_only, %d left out on purpose' % (len(tree), len(left_out)))

# 3 and 4: support manifest
man = rows(MAN)
by_class = {}
mfiles = set()
n_ok = 0
for r in man:
    by_class[r['class']] = by_class.get(r['class'], 0) + 1
    cls = r['class']
    if cls in ('PUSH-VERBATIM', 'PUSH-REDACTED', 'TIER-BUILD'):
        rel = r['repo_path']
        mfiles.add(rel)
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            p('manifest file missing: ' + rel)
        elif sha(full) != r['repo_sha256']:
            p('manifest file changed: ' + rel)
        else:
            n_ok += 1
    elif cls == 'DUPLICATE-OF-REPO':
        rel = r['duplicate_of']
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            p('duplicate target missing: ' + rel)
        elif sha(full) != r['repo_sha256']:
            p('duplicate target differs: ' + rel)
        else:
            n_ok += 1
print('manifest: %d rows (%s); %d copies or duplicate targets match' % (
    len(man), ', '.join('%s %d' % kv for kv in sorted(by_class.items())), n_ok))
arch = set()
for dp, dn, fn in os.walk(os.path.join(ROOT, 'support', 'archive')):
    for f in fn:
        arch.add(os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, '/'))
for rel in sorted(arch - mfiles):
    p('archive file not in manifest: ' + rel)
print('archive: %d files, all listed' % len(arch) if not (arch - mfiles) else 'archive: %d files' % len(arch))

# 5: privacy split
gi = os.path.join(ROOT, '.gitignore')
if not os.path.isfile(gi) or 'support/local_only/' not in open(gi, encoding='utf-8').read():
    p('.gitignore does not ignore support/local_only/')
try:
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', 'support/local_only'], capture_output=True, text=True)
    if out.returncode == 0:
        if out.stdout.strip():
            p('tracked files under support/local_only: %d' % len(out.stdout.split()))
        else:
            print('git: no file under support/local_only is tracked')
    else:
        notes.append('git check skipped (not a git work tree)')
except OSError:
    notes.append('git check skipped (git not found)')

# 6: working copy only
if os.path.isfile(os.path.join(ROOT, LOCAL_MAN)):
    loc = rows(LOCAL_MAN)
    bad = 0
    for r in loc:
        full = os.path.join(ROOT, r['repo_path'])
        if not os.path.isfile(full) or sha(full) != r['repo_sha256']:
            bad += 1
            p('local file missing or changed: ' + r['repo_path'])
    print('local manifest: %d rows, %d problems' % (len(loc), bad))
else:
    notes.append('local manifest not present (a clone does not carry the withheld files)')

for n in notes:
    print('note:', n)
print('RESULT:', 'all checks passed' if not problems else '%d problem(s)' % len(problems))
sys.exit(1 if problems else 0)
