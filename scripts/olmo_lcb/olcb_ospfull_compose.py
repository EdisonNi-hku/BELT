from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

RUNS = [f'osp-lam{lam}-s{s}-fb' for lam in ('2e-6', '5e-6', '1e-5', '2e-5', '4e-5') for s in (0, 1, 2)]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def complete(d: Path, bank_sha: str):
    try:
        s = json.loads((d / 'summary.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False, None
    n = s['config']['n_select_chains']
    rows = sum(1 for line in open(d / 'select_scored.jsonl') if line.strip())
    prov = s['provenance']
    ok = rows == n and prov['chains']['sha256'] == bank_sha
    return ok, {'n_select_chains': n, 'rows': rows, 'chains_sha256': prov['chains']['sha256'],
                'osp_checkpoint_sha256': prov['osp_checkpoint']['sha256'],
                'device': prov.get('device'),
                'select_sha256': sha(d / 'select_scored.jsonl'), 'path': str(d.resolve())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--bank-json', required=True)
    ap.add_argument('--native-root', required=True, type=Path)
    ap.add_argument('--gh200-root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    bank_sha = json.loads(Path(a.bank_json).read_text())['outputs']['chains']['sha256']
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {'schema': 'olcb-ospfull-compose/v1', 'pool': a.pool, 'bank_chains_sha256': bank_sha,
           'rule': 'native card (Euler, delivered ext74 card) if complete, else GH200 with --allow-device-mismatch',
           'runs': {}}
    missing = []
    for r in RUNS:
        nat_ok, nat = complete(a.native_root / r, bank_sha)
        gh_ok, gh = complete(a.gh200_root / r, bank_sha)
        src, info = ('native', nat) if nat_ok else (('gh200', gh) if gh_ok else (None, None))
        if src is None:
            missing.append(r)
            continue
        if nat_ok and gh_ok and nat['osp_checkpoint_sha256'] != gh['osp_checkpoint_sha256']:
            raise SystemExit(f'{r}: native and GH200 scorings used different checkpoints')
        link = a.out / r
        if link.is_symlink() or link.exists():
            if os.readlink(link) != info['path']:
                raise SystemExit(f'{link} already points at {os.readlink(link)}')
        else:
            link.symlink_to(info['path'], target_is_directory=True)
        doc['runs'][r] = {'source': src, **info, 'native_available': nat_ok, 'gh200_available': gh_ok}
    if missing:
        raise SystemExit(f'no complete scoring for {missing}')
    (a.out / 'compose.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({'pool': a.pool, 'native': sum(v['source'] == 'native' for v in doc['runs'].values()),
                      'gh200': sum(v['source'] == 'gh200' for v in doc['runs'].values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
