from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import D53, REB, load_frontier_lib, read_all, sha256, write_results

N = load_frontier_lib()
TOLS = (0, 1, 2)
OUT = REB / 'dev_selected'
INP = OUT / 'inputs'
METHODS = [('BELT-B', 'B-R'), ('BELT-H', 'H-R'), ('BELT-B-1', 'B-T'), ('BELT-H-1', 'H-T'),
           ('OS-Pruner full', 'OSP'), ('OS-Pruner single', 'OSP-1')]
NOT_AVAILABLE = {'TC': 'no dev-160 rows in the E1 roots (test rows only)', 'ROM': 'no dev-160 rows in the E1 roots (test rows only)',
                 'OSP+T': 'no dev rows exist (b9b11 threshold expansion is test-only)',
                 'OSP-1+T': 'never produced (Task 3, Entry 9) and no dev rows'}
SETTINGS = [
    ('Qwen ID', 'qwen_id', {'fb_ri': 'e1a_v2', 'fb': 'e1a_v2fb'}, 'math dev-160'),
    ('Qwen OOD', 'qwen_ood', {'fb_ri': 'e1a_v2', 'fb': 'e1a_v2fb'}, 'math dev-160'),
    ('LCB Pre', 'lcb_pre', {'fb_ri': 'b8_v2ri', 'fb': 'b8_v2fb'}, 'math dev-160 (source side)'),
    ('LCB Post', 'lcb_post', {'fb_ri': 'b8_v2ri', 'fb': 'b8_v2fb'}, 'math dev-160 (source side)'),
    ('OLMo', 'olmo_ood', {'fb_ri': 'e1a_olmo_v1', 'fb': 'e1a_olmo_v1'}, 'OLMo dev-160'),
]


def load_scope(root, scope):
    idx = json.loads((root / f'{scope}.json').read_text())
    z = np.load(root / f'{scope}.npz', allow_pickle=True)
    rec = {'dataset': z['dataset'], 'N': z['N'], 'Nd': z['Nd'], 'methods': {}}
    for m, by_seed in idx['methods'].items():
        for s, v in by_seed.items():
            rec['methods'].setdefault(m, {})[int(s)] = {'names': v['names'], 'X': z[f'{m}|{s}|X'],
                                                        'D': z[f'{m}|{s}|D'] if v['has_dev'] else None}
    return rec


def evaluate(choice, X, nat, ds):
    vv = sum(w * (nat if i == N.NATURAL else X[i]) for i, w in choice)
    dsn = sorted(set(ds.tolist()))
    acc = lambda a, n: (float(n[:, 0].mean()) - float(a[:, 0].mean())) * 100
    out = {'drop_micro': acc(vv, nat), 'saving_micro': N.summarize(vv, nat, 'token_saving', 0),
           'drop_macro': float(np.mean([acc(vv[ds == d], nat[ds == d]) for d in dsn])),
           'saving_macro': float(np.mean([N.summarize(vv[ds == d], nat[ds == d], 'token_saving', 0) for d in dsn]))}
    return out


def select(D, Nd, tol):
    micro, nm = D.mean(1), Nd.mean(0)
    front = N.frontier(micro, nm)
    choice, status = N.read(front, micro, nm, 'parity', tol)
    assert choice is not None, status
    return choice


def ms(v):
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else None)


def printed_cells():
    out = {}
    for fam in ('fb_ri', 'fb'):
        for r in csv.DictReader((D53 / f'v2/paper_cells/cells_{fam}.csv').open()):
            if r['table'] == 'tab:allbaselines' and r['v2_status'] == 'complete':
                p, lv, scope, agg = r['column'].split()
                out[(fam, 'Qwen ' + scope, r['row'], int(lv))] = float(r['v2_full'])
            if fam == 'fb_ri' and r['table'] == 'tab:olmo-main' and r['column'].startswith('sav ') and r['v2_status'] == 'complete':
                lv = int(r['column'].split()[1][0])
                for f in ('fb_ri', 'fb'):
                    out[(f, 'OLMo', r['row'], lv)] = float(r['v2_full'])
        key = {'fb_ri': 'v2ri', 'fb': 'v2fb'}[fam]
        for r in csv.DictReader((D53 / f'v2/items_d28/code/k1/e1/{key}/cross_reading_cells.csv').open()):
            if r['protocol'] == 'parity' and r['same_set']:
                st = {'lcb_pre': 'LCB Pre', 'lcb_post': 'LCB Post'}[r['scope']]
                out.setdefault((fam, st, r['method'], int(r['level'])), {})[int(r['seed'])] = float(r['same_set'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    printed = printed_cells()
    rows, g1, g2, inputs = [], [], [], {}
    for setting, scope, dirs, devdesc in SETTINGS:
        for fam, sub in dirs.items():
            root = INP / sub
            for f in (f'{scope}.npz', f'{scope}.json'):
                inputs[str((root / f).relative_to(a.out))] = sha256(root / f)
            rec = load_scope(root, scope)
            Nt, ds, Nd = rec['N'], np.asarray(rec['dataset']), rec['Nd']
            for m, short in METHODS:
                if m not in rec['methods']:
                    continue
                per_seed = {}
                for s, v in sorted(rec['methods'][m].items()):
                    assert v['D'] is not None, (setting, m, s)
                    X = v['X']
                    test = read_all(X, Nt, ds)
                    for tol in TOLS:
                        t = test[f'parity|{tol:g}']
                        tt = evaluate(select(X, Nt, tol), X, Nt, ds)
                        g2.append({'setting': setting, 'family': fam, 'method': short, 'seed': s, 'tol': tol,
                                   'err': max(abs(tt['saving_micro'] - t['micro']), abs(tt['saving_macro'] - t['macro'])),
                                   'same_members': [list(x) for x in t['members']] == [list(x) for x in select(X, Nt, tol)]})
                        ch = select(v['D'], Nd, tol)
                        dev = evaluate(ch, v['D'], Nd, np.zeros(len(Nd), dtype=int))
                        r = evaluate(ch, X, Nt, ds)
                        per_seed.setdefault(tol, []).append({
                            'seed': s, 'members': [('natural' if i == N.NATURAL else v['names'][i], w) for i, w in ch],
                            'dev_drop': dev['drop_micro'], 'dev_saving': dev['saving_micro'], **r,
                            'test_curve_saving_micro': t['micro'], 'test_curve_saving_macro': t['macro']})
                for tol, lst in per_seed.items():
                    pc = printed.get((fam, setting, m, tol))
                    agg = 'micro' if setting.startswith('LCB') else 'macro'
                    if isinstance(pc, dict):
                        for x in lst:
                            g1.append({'setting': setting, 'family': fam, 'method': short, 'tol': tol, 'seed': x['seed'],
                                       'err': abs(x[f'test_curve_saving_{agg}'] - pc[x['seed']])})
                    elif pc is not None:
                        g1.append({'setting': setting, 'family': fam, 'method': short, 'tol': tol, 'seed': 'mean',
                                   'err': abs(statistics.fmean(x[f'test_curve_saving_{agg}'] for x in lst) - pc)})
                    rows.append({'setting': setting, 'family': fam, 'method': short, 'method_id': m, 'tol': tol,
                                 'dev': devdesc, 'n_seeds': len(lst), 'per_seed': lst,
                                 **{k: ms([x[k] for x in lst]) for k in ('drop_macro', 'drop_micro', 'saving_macro',
                                                                         'saving_micro', 'dev_drop', 'dev_saving',
                                                                         'test_curve_saving_macro', 'test_curve_saving_micro')},
                                 'seeds_drop_over_tol_macro': sum(x['drop_macro'] > tol + 1e-9 for x in lst)})
    g1_bad = [x for x in g1 if x['err'] > 1e-8]
    g2_bad = [x for x in g2 if x['err'] > 1e-8]
    gates = {'G1_printed_v2_parity_cells': {'n': len(g1), 'max_err': max(x['err'] for x in g1), 'n_fail': len(g1_bad),
                                            'fails': g1_bad[:10]},
             'G2_dev_equals_test_reproduces_parity': {'n': len(g2), 'max_err': max(x['err'] for x in g2),
                                                      'n_fail': len(g2_bad), 'fails': g2_bad[:10],
                                                      'all_same_members': all(x['same_members'] for x in g2)}}
    want = {(f, st, m, t) for (f, st, m, t) in printed if any(m == mm for mm, _ in METHODS)}
    seen = {(x['family'], x['setting'], dict((s, l) for l, s in METHODS)[x['method']], x['tol']) for x in g1}
    gates['G1_printed_v2_parity_cells']['uncompared'] = sorted(map(str, want - seen))
    md = render(rows, gates)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'dev_selected_tolerance.md').write_text(md)
    readme = ['# Task 4 Step 2 -- dev-selected tolerance operating points', '',
              'Producer: `scripts/belt_ext/d53/reb/dev_selected_tolerance.py` (see its docstring for the protocol and gates).',
              'Output: `dev_selected_tolerance.md`, `RESULTS.json`.', '', '## Inputs (sha256)', '',
              '`inputs/` is kept out of git (pulled from Euler; sources below).', '',
              '| file | sha256 |', '|---|---|'] + [f'| `{k}` | `{v}` |' for k, v in sorted(inputs.items())] + [
              '', 'Euler sources: `e1a_v2` = `$P/iclr/runs/belt_ext/d53/v2/v2both/e1a/v2/data`, `e1a_v2fb` = `.../e1a/v2fb/data`, '
              '`b8_v2ri` / `b8_v2fb` = `$P/iclr/runs/belt_ext/d53/b8_lcb_closeout_20261003/{v2ri,v2fb}/data`, `e1a_olmo_v1` = '
              '`$P/iclr/runs/belt_ext/d53/items_d28/e1a/olmo_v1/data` (P = /cluster/project/sachan/njingwei).',
              '', 'Printed-cell references (G1): `v2/paper_cells/cells_{fb_ri,fb}.csv` '
              f'({sha256(D53 / "v2/paper_cells/cells_fb_ri.csv")[:16]}…, {sha256(D53 / "v2/paper_cells/cells_fb.csv")[:16]}…), '
              '`v2/items_d28/code/k1/e1/{v2ri,v2fb}/cross_reading_cells.csv`.', '']
    (a.out / 'README.md').write_text('\n'.join(readme) + '\n')
    write_results(a.out, {'protocol': 'dev-selected parity (N.frontier / N.read on dev-160), realized on test',
                          'tolerances_pp': list(TOLS), 'not_available': NOT_AVAILABLE, 'gates': gates,
                          'inputs_sha256': inputs, 'rows': rows},
                  extra_files=[a.out / 'dev_selected_tolerance.md', a.out / 'README.md'])
    print(json.dumps({k: {kk: vv for kk, vv in g.items() if kk != 'fails'} for k, g in gates.items()}))
    if g1_bad or g2_bad or gates['G1_printed_v2_parity_cells']['uncompared']:
        raise SystemExit('gate failure: see RESULTS.json gates')


def fmt(m, sd, d=2):
    return f'{m:.{d}f}' if sd is None else f'{m:.{d}f} ± {sd:.{d}f}'


def render(rows, gates):
    L = ['# Dev-selected tolerance operating points (Task 4 Step 2)', '',
         'Protocol: on the dev-160 rows, the dev frontier (`N.frontier`) and the parity reading (`N.read`, '
         '"cheapest frontier point whose dev accuracy reaches dev natural - tol", adjacent-member mixture) pick the '
         'operating point; the same members / weights are evaluated on the test population. **Realized drop** = test '
         'natural accuracy - test accuracy (pp; negative = above natural); **saving** = token saving vs test natural (%). '
         '"test-curve" = the paper\'s test-selected parity reading at the same tolerance (the printed v2 cell). '
         'Mean ± SD over the three seeds; Qwen / OLMo macro (the printed aggregation), LCB micro = macro (one dataset). '
         '"> tol" = seeds whose realized macro drop exceeds the tolerance.', '',
         'Dev sets: Qwen ID / OOD and LCB Pre / Post = math dev-160 (LCB: source-side selection, as the paper\'s '
         'utility tables); OLMo = OLMo dev-160.', '',
         'Not available: ' + '; '.join(f'**{k}** ({v})' for k, v in NOT_AVAILABLE.items()) + '.', '',
         f"Gates: G1 printed v2 parity cells {gates['G1_printed_v2_parity_cells']['n'] - gates['G1_printed_v2_parity_cells']['n_fail']}"
         f"/{gates['G1_printed_v2_parity_cells']['n']} (max err {gates['G1_printed_v2_parity_cells']['max_err']:.1e}); "
         f"G2 dev := test reproduces the parity reading {gates['G2_dev_equals_test_reproduces_parity']['n'] - gates['G2_dev_equals_test_reproduces_parity']['n_fail']}"
         f"/{gates['G2_dev_equals_test_reproduces_parity']['n']} (max err {gates['G2_dev_equals_test_reproduces_parity']['max_err']:.1e}).", '']
    for fam, title in (('fb_ri', '-fb-ri (random init, main line)'), ('fb', '-fb (ReProbe init, analysis only)')):
        L += [f'## Family {title}', '']
        for setting, *_ in SETTINGS:
            sub = [r for r in rows if r['family'] == fam and r['setting'] == setting]
            if not sub:
                continue
            agg = 'micro' if setting.startswith('LCB') else 'macro'
            L += [f'### {setting}', '',
                  '| method | tol | realized drop (pp) | > tol | realized saving (%) | test-curve saving (%) | dev drop (pp) | dev saving (%) |',
                  '|---|---|---|---|---|---|---|---|']
            for r in sub:
                L.append(f"| {r['method']} | {r['tol']} | {fmt(*r['drop_' + agg])} | {r['seeds_drop_over_tol_macro']}/{r['n_seeds']} | "
                         f"{fmt(*r['saving_' + agg])} | {fmt(*r['test_curve_saving_' + agg])} | {fmt(*r['dev_drop'])} | {fmt(*r['dev_saving'])} |")
            L.append('')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
