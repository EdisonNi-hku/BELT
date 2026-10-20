from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

REPO = Path(__file__).resolve().parents[4]
D53 = REPO / 'results/belt_ext/natural_fallback/d53'
REB = D53 / 'reb'
X5_INPUTS = D53 / 'nes_frontier_v1/inputs'
FRONTIER_LIB = D53 / 'osp1_extension_v1/x19_summary.source.py'
_N = None


def load_frontier_lib():
    global _N
    if _N is None:
        spec = importlib.util.spec_from_file_location('native', FRONTIER_LIB)
        N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)
        _N = N
    return _N


N = load_frontier_lib()
READINGS = [('parity', [0, 1, 2], 'token_saving'), ('mixture', [.3, .4, .5, .6], 'accuracy'),
            ('price', N.PRICES, 'utility')]
READING_KEYS = [f'{p}|{lv:g}' for p, levels, _ in READINGS for lv in levels]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_all(points, natural, datasets):
    X = np.asarray(points, dtype=float); nat = np.asarray(natural, dtype=float)
    datasets = np.asarray(datasets); dsnames = sorted(set(datasets.tolist()))
    assert X.ndim == 3 and X.shape[2] == 3 and nat.shape == X.shape[1:], (X.shape, nat.shape)
    micro = X.mean(1); nm = nat.mean(0)
    front = N.frontier(micro, nm)
    row = {}
    for protocol,levels,metric in READINGS:
        for level in levels:
            choice,status=N.read(front,micro,nm,protocol,level)
            vv=None if choice is None else sum(w*(nat if i==N.NATURAL else X[i]) for i,w in choice)
            key=f'{protocol}|{level:g}'
            row[key]=dict(status=status,micro=None if vv is None else N.summarize(vv,nat,metric,level),
                          macro=None if vv is None else float(np.mean([N.summarize(vv[datasets==d],nat[datasets==d],metric,level) for d in dsnames])),
                          members=choice)
            if vv is not None and protocol=='parity':
                assert vv[:,0].mean()>=nm[0]-level/100-1e-10
            if vv is not None and protocol=='mixture':
                assert vv[:,2].mean()<=level*nm[2]+1e-9
    return row


def x5_family(arm, seed, inputs=X5_INPUTS, pools=('math_ood_report', 'ood2_math_report')):
    import sys
    if str(inputs) not in sys.path:
        sys.path.insert(0, str(inputs))
    import replay_npz_task as A
    roster = A.roster(Path(inputs) / 'test_roster.jsonl')
    shards = {}
    for pool in pools:
        path = Path(inputs) / f'x5_{pool}_s{seed}.npz'
        side = json.loads(path.with_suffix('.json').read_text())
        assert side['columns'] == ['strict_accuracy', 'loose_accuracy', 'thinking_tokens', 'total_tokens']
        assert side['reproduces_archived'] is True and not side['discrepancies']
        shards[pool] = A.load_arm(path, arm)
    idx = {p: {str(q): j for j, q in enumerate(v['pool_qkeys'])} for p, v in shards.items()}
    points = np.stack([shards[r['origin_pool']]['pool'][:, idx[r['origin_pool']][r['source_qkey']], :]
                       for r in roster], axis=1)
    natural = np.stack([shards[r['origin_pool']]['pool_natural'][idx[r['origin_pool']][r['source_qkey']]]
                        for r in roster])
    assert np.isfinite(points).all() and np.isfinite(natural).all()
    datasets = np.array([r['dataset_id'] for r in roster])
    return points[:, :, [0, 2, 3]], natural[:, [0, 2, 3]], datasets, shards


def write_results(out_dir, obj, extra_files=()):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'RESULTS.json').write_text(json.dumps(obj, indent=2, default=float) + '\n')
    names = ['RESULTS.json', *[str(Path(f).relative_to(out)) if Path(f).is_absolute() else str(f) for f in extra_files]]
    (out / 'SHA256SUMS').write_text(''.join(f'{sha256(out / n)}  {n}\n' for n in sorted(set(names))))
    return out / 'RESULTS.json'
