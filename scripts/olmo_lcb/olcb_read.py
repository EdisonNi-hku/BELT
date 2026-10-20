from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path

NATURAL = 'natural'
TOLERANCES = ['0pp', '1pp', '2pp']
BUDGETS = ['0.3', '0.4', '0.5', '0.6']
PRICES = ['2e-06', '5e-06', '1e-05', '2e-05', '5e-05']
READ7 = [('sav', lv) for lv in TOLERANCES] + [('acc', lv) for lv in BUDGETS]
POOLS = [('ood2_code_report', 'Pre', 549), ('lcb_post_f2048', 'Post', 416)]
SEEDS = (0, 1, 2)
VARIANTS = {'H-R': 'BELT-H', 'H-T': 'BELT-H-1', 'B-R': 'BELT-B', 'B-T': 'BELT-B-1'}
COMPARATORS = ('OSP-full', 'OSP-1')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_export(path, split='pool'):
    e = json.loads(gzip.open(path, 'rt').read())
    assert e['columns'] == ['strict_accuracy', 'thinking_tokens', 'total_tokens'], path
    return {(r['method'], r['member']): r['q'] for r in e['rows'] if r['split'] == split}


class Pop:

    def __init__(self, natural, dataset_of, datasets):
        self.natural = natural
        self.qkeys = sorted(natural)
        self.n = len(self.qkeys)
        self.datasets = list(datasets)
        self.by_dataset = {d: [q for q in self.qkeys if dataset_of[q] == d] for d in self.datasets}
        assert sum(len(v) for v in self.by_dataset.values()) == self.n
        self.nat_micro = (sum(natural[q][0] for q in self.qkeys) / self.n, sum(natural[q][2] for q in self.qkeys) / self.n)

    def micro(self, q):
        return (sum(q[k][0] for k in self.qkeys) / self.n, sum(q[k][2] for k in self.qkeys) / self.n)

    def frontier(self, tables):
        pts = [(self.micro(q)[1], self.micro(q)[0], name) for name, q in tables.items()] + [(self.nat_micro[1], self.nat_micro[0], NATURAL)]
        pts.sort(key=lambda p: (p[0], -p[1], p[2]))
        front, best = [], -1.0
        for t, a, n in pts:
            if a > best:
                front.append((t, a, n)); best = a
        return front

    def read(self, front, tables, kind, lv):
        nat_micro = self.nat_micro
        if kind == 'sav':
            target = nat_micro[0] - int(lv[0]) / 100
            if front[0][1] >= target - 1e-12:
                return [(front[0][2], 1.0)]
            for (t0, a0, n0), (t1, a1, n1) in zip(front, front[1:]):
                if a0 < target <= a1:
                    w1 = (target - a0) / (a1 - a0)
                    return [(n0, 1 - w1), (n1, w1)]
            return None
        if kind == 'acc':
            cost = float(lv) * nat_micro[1]
            if cost < front[0][0] - 1e-9:
                return None
            if cost >= front[-1][0] - 1e-9:
                return [(front[-1][2], 1.0)]
            for (t0, a0, n0), (t1, a1, n1) in zip(front, front[1:]):
                if t0 - 1e-9 <= cost <= t1 + 1e-9:
                    if abs(t1 - t0) < 1e-9:
                        return [(n1, 1.0)]
                    w1 = (cost - t0) / (t1 - t0)
                    return [(n0, 1 - w1), (n1, w1)]
            raise AssertionError('unreachable')
        lam = float(lv)
        best_name, best_u = NATURAL, nat_micro[0] - lam * nat_micro[1]
        for name, q in tables.items():
            a, t = self.micro(q); u = a - lam * t
            if u > best_u + 1e-15:
                best_name, best_u = name, u
        return [(best_name, 1.0)]

    def mixed_table(self, tables, mix):
        natural = self.natural
        return {k: [sum(w * (natural[k][i] if n == NATURAL else tables[n][k][i]) for n, w in mix) for i in range(3)] for k in self.qkeys}

    def slice_stats(self, q, lam=None):
        natural, qkeys, nd = self.natural, self.qkeys, len(self.datasets)
        acc_ds, sav_ds, u_ds = [], [], []
        for d in self.datasets:
            ks = self.by_dataset[d]
            acc_ds.append(100 * sum(q[k][0] for k in ks) / len(ks))
            sav_ds.append(100 * (1 - sum(q[k][2] for k in ks) / sum(natural[k][2] for k in ks)))
            if lam is not None:
                u_ds.append(sum(q[k][0] - lam * q[k][2] for k in ks) / len(ks))
        out = {'acc': {'micro': 100 * sum(q[k][0] for k in qkeys) / self.n, 'macro': sum(acc_ds) / nd},
               'sav': {'micro': 100 * (1 - sum(q[k][2] for k in qkeys) / sum(natural[k][2] for k in qkeys)), 'macro': sum(sav_ds) / nd}}
        if lam is not None:
            out['utility'] = {'micro': sum(q[k][0] - lam * q[k][2] for k in qkeys) / self.n, 'macro': sum(u_ds) / nd}
        return out


def msd(vals):
    m = sum(vals) / len(vals)
    return m, ((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None)


class Reader:

    def __init__(self, pop, families):
        self.pop, self.families = pop, families
        self.fronts = {}

    def reading(self, row, kind, lv):
        pop = self.pop
        per_seed, detail = {}, {}
        for s, tables in self.families[row].items():
            if (row, s) not in self.fronts:
                self.fronts[(row, s)] = pop.frontier(tables)
            mix = pop.read(self.fronts[(row, s)], tables, kind, lv)
            if mix is None:
                return {'row': row, 'kind': kind, 'level': lv, 'infeasible_seed': s}
            st = pop.slice_stats(pop.mixed_table(tables, mix), float(lv) if kind == 'price' else None)
            per_seed[s] = st
            detail[s] = {'mixture': [{'member': n, 'weight': w} for n, w in mix]}
        key = 'acc' if kind == 'acc' else 'sav' if kind == 'sav' else 'utility'
        stats = {agg: msd([per_seed[s][key][agg] for s in per_seed]) for agg in ('micro', 'macro')}
        return {'row': row, 'kind': kind, 'level': lv, 'n_seeds': len(per_seed),
                'micro': stats['micro'][0], 'macro': stats['macro'][0],
                'seed_sd': {'micro': stats['micro'][1], 'macro': stats['macro'][1]},
                'per_seed': {str(s): {'micro': per_seed[s][key]['micro'], 'macro': per_seed[s][key]['macro'], **detail[s]} for s in per_seed}}

    def all_cells(self, rows):
        out = []
        for row in rows:
            for kind, levels in (('sav', TOLERANCES), ('acc', BUDGETS), ('price', PRICES)):
                if kind == 'sav' and row == 'Fixed cap':
                    continue
                for lv in levels:
                    out.append(self.reading(row, kind, lv))
        return out


def math_families(shadow: Path):
    EV = shadow / 'evidence/olmo_20260921_1025'
    EXPORT = EV / 'baselines_math_ood_1025.json.gz'
    SEED_DIR, OSP_DIR, X13 = EV / 'seeds', EV / 'osp_seeds', shadow / 'evidence/x13_20260922/exports3'
    F = {
        'Fixed cap': {0: (EXPORT, ['cap@dev160'])},
        'CoDE-Stop': {0: (EXPORT, ['CoDE@olmo3conf-gre'])},
        'DEER': {0: (EXPORT, ['DEER@olmo3conf-gre'])},
        'TC': {s: (SEED_DIR / f'export_tc-s{s}.json.gz', [f'TCLAM@tc-L18-w{w}-s{s}' for w in (1, 5, 10)]) for s in SEEDS},
        'ROM': {0: (EXPORT, ['ROM_tau@cfc-L18']), 1: (SEED_DIR / 'export_rom-s1.json.gz', ['ROM_tau@cfc-L18-s1']), 2: (SEED_DIR / 'export_rom-s2.json.gz', ['ROM_tau@cfc-L18-s2'])},
        'OSP-1': {0: (EXPORT, ['OSP-single-threshold@dev160']), 1: (OSP_DIR / 'export_osp-s1.json.gz', ['OSP-single-threshold@dev160-s1']), 2: (OSP_DIR / 'export_osp-s2.json.gz', ['OSP-single-threshold@dev160-s2'])},
        'BELT-B': {0: (EXPORT, ['LB_QVRS@qvrs-lbdev']), 1: (SEED_DIR / 'export_reader_s1.json.gz', ['LB_QVRS@qvrs-lbdev-s1']), 2: (SEED_DIR / 'export_reader_s2.json.gz', ['LB_QVRS@qvrs-lbdev-s2'])},
        'BELT-H': {0: (EXPORT, ['LB_H@qvrs-lbdev']), 1: (SEED_DIR / 'export_reader_s1.json.gz', ['LB_H@qvrs-lbdev-s1']), 2: (SEED_DIR / 'export_reader_s2.json.gz', ['LB_H@qvrs-lbdev-s2'])},
        'BELT-B-1': {s: (X13 / f'exports3-s{s}.json.gz', [f'LB_QVRS_1@olmo-thr-s{s}']) for s in SEEDS},
        'BELT-H-1': {s: (X13 / f'exports3-s{s}.json.gz', [f'LB_H_1@olmo-thr-s{s}']) for s in SEEDS},
        'OSP-full': {s: (EV / 'osp_full' / f'export_osp_full-s{s}.json.gz', ['OSP@dev160']) for s in SEEDS},
        'OSP+T': {s: (EV / 'osp_plus_t' / f'export_osp_plus_t-s{s}.json.gz', ['OSP+T@dev160']) for s in SEEDS},
    }
    expected = {'Fixed cap': 8, 'CoDE-Stop': 75, 'DEER': 18, 'TC': 150, 'ROM': 21, 'OSP-1': 13, 'BELT-B': 221, 'BELT-H': 221,
                'BELT-B-1': 13, 'BELT-H-1': 13, 'OSP-full': 5, 'OSP+T': 70}
    return EXPORT, F, expected


def build_families(F, expected, natural, split='pool'):
    fams, sources = {}, {}
    for row, per in F.items():
        fams[row] = {}
        for s, (path, methods) in per.items():
            rows = load_export(path, split)
            if split == 'pool':
                assert rows[(NATURAL, NATURAL)] == natural, ('natural anchor differs', path)
            mem = {f'{m}|{k}': q for (m, k), q in rows.items() if m in methods}
            assert len(mem) == expected[row], (row, s, len(mem))
            for q in mem.values():
                assert set(q) == set(natural), (row, s, 'incomplete member')
            fams[row][s] = mem
            sources[str(path)] = None
    return fams, sources


def gate_math(shadow: Path):
    EXPORT, F, expected = math_families(shadow)
    roster = shadow / 'evidence/reporting_20260920/test_roster.jsonl'
    dataset_of = {json.loads(l)['source_qkey']: json.loads(l)['dataset_id'] for l in roster.read_text().splitlines()}
    natural = load_export(EXPORT)[(NATURAL, NATURAL)]
    pop = Pop(natural, dataset_of, ['math', 'olympiadbench', 'gaokao', 'aime', 'amc'])
    assert pop.n == 1025 and [len(pop.by_dataset[d]) for d in pop.datasets] == [295, 366, 191, 98, 75]
    fams, _ = build_families(F, expected, natural)
    rows = ['Fixed cap', 'CoDE-Stop', 'DEER', 'TC', 'ROM', 'OSP-1', 'OSP-full', 'OSP+T', 'BELT-B-1', 'BELT-H-1', 'BELT-B', 'BELT-H']
    mine = {(c['row'], c['kind'], c['level']): c for c in Reader(pop, fams).all_cells(rows)}
    want = {}
    for f in ('OLMO_OPERATING_RENDER_RECEIPT.json', 'OLMO_RENDER_RECEIPT.json'):
        for c in json.loads((shadow / 'generated/framework' / f).read_text())['cells']:
            want[(c['row'], c['kind'], c['level'])] = c
    assert set(mine) == set(want), sorted(set(mine) ^ set(want))
    err, nvals = 0.0, 0
    for k, w in want.items():
        m = mine[k]
        for agg in ('micro', 'macro'):
            pairs = [(m[agg], w[agg]), (m['seed_sd'][agg], w['seed_sd'][agg])] + [(m['per_seed'][s][agg], w['per_seed'][s][agg]) for s in w['per_seed']]
            for x, y in pairs:
                if x is None or y is None:
                    assert x is None and y is None, (k, agg)
                    continue
                err = max(err, abs(x - y)); nvals += 1
        assert [d['member'] for d in w['per_seed']['0']['mixture']] == [d['member'] for d in m['per_seed']['0']['mixture']], k
    n_pre = sum(1 for k in want if k[0] != 'OSP+T')
    return {'cells': len(want), 'cells_before_osp_plus_t': n_pre, 'values_compared': nvals, 'max_abs_diff': err,
            'ok': err == 0.0}, pop, fams, mine


def st(v):
    return statistics.mean(v), statistics.stdev(v)


def verdict(a, b):
    (ma, sa), (mb, sb) = st(a), st(b)
    d = ma - mb; t = max(sa, sb)
    return d, ('T' if abs(d) <= t else ('W' if d > 0 else 'L'))


def comp(a, b):
    if a is None or b is None:
        return {'verdict': 'NF'}
    d, v = verdict(a, b)
    o = {'diff': d, 'verdict': v, 'belt': list(st(a)), 'comp': list(st(b))}
    if v in 'WL':
        o['robust'] = all(verdict(a, [x for j, x in enumerate(b) if j != i])[1] == v for i in range(3)) and \
                      all(verdict([x for j, x in enumerate(a) if j != i], b)[1] == v for i in range(3))
    return o


def per_seed_list(cells, row, kind, lv, agg):
    c = cells.get((row, kind, lv))
    if c is None or 'per_seed' not in c or len(c['per_seed']) != 3:
        return None
    return [c['per_seed'][str(s)][agg] for s in SEEDS]


def wtl(cells, agg, comparators=COMPARATORS):
    out = {}
    for var, row in VARIANTS.items():
        for cmp_ in comparators:
            cs = [comp(per_seed_list(cells, row, k, lv, agg), per_seed_list(cells, cmp_, k, lv, agg)) for k, lv in READ7]
            cnt = {x: sum(c['verdict'] == x for c in cs) for x in ('W', 'T', 'L', 'NF')}
            rob = {x: sum(c['verdict'] == x and c.get('robust', False) for c in cs) for x in 'WL'}
            out[f'{var}|{cmp_}'] = {'counts': cnt, 'robust': rob, 'readings': {f'{k}|{lv}': c for (k, lv), c in zip(READ7, cs)}}
    return out


def gate_wtl(math_cells, task9_result):
    got = wtl(math_cells, 'macro')
    t9 = json.loads(Path(task9_result).read_text())['cells']
    n, bad = 0, []
    for key, c in t9.items():
        var, cmp_, kind, lv = key.split('|')
        if kind == 'price':
            continue
        g = got[f'{var}|{cmp_}']['readings'][f'{kind}|{lv}']
        n += 1
        if g['verdict'] != c['verdict'] or abs(g['diff'] - c['diff']) > 0 or g.get('robust') != c.get('robust'):
            bad.append(key)
    paper = {'H-R': ((7, 0, 0), (6, 0)), 'H-T': ((5, 1, 1), (3, 0)), 'B-R': ((6, 1, 0), (6, 0)), 'B-T': ((5, 1, 1), (5, 1))}
    for var, (wtl_, rob) in paper.items():
        g = got[f'{var}|OSP-1']
        if (g['counts']['W'], g['counts']['T'], g['counts']['L']) != wtl_ or (g['robust']['W'], g['robust']['L']) != rob:
            bad.append(f'paper {var} vs OSP-1')
    return {'task9_readings': n, 'paper_osp1_rows': len(paper), 'mismatches': bad, 'ok': not bad}


def micro_u(q, lam):
    return sum(v[0] - lam * v[2] for v in q.values()) / len(q)


def select(dev, test, natural_dev, natural_test, lam, test_pop):
    best_name, best = NATURAL, micro_u(natural_dev, lam)
    for m in sorted(dev):
        u = micro_u(dev[m], lam)
        if u > best:
            best_name, best = m, u
    q = natural_test if best_name == NATURAL else test[best_name]
    st_ = test_pop.slice_stats(q, lam)['utility']
    return best_name, st_['micro'], st_['macro']


def devsel(fam_dev, fam_test, natural_dev, natural_test, pop, rows):
    res = {}
    for row in rows:
        for p in PRICES:
            per = {s: select(fam_dev[row][s], fam_test[row][s], natural_dev, natural_test, float(p), pop) for s in SEEDS}
            for agg, i in (('micro', 1), ('macro', 2)):
                v = [per[s][i] for s in SEEDS]
                res[f'{row}|{agg}|{p}'] = {'mean': statistics.mean(v), 'sd': statistics.stdev(v), 'per_seed': v,
                                           'chosen': [per[s][0] for s in SEEDS]}
    return res


def devsel_families(F, expected, rows):
    out = {}
    for row in rows:
        out[row] = {}
        for s, (path, methods) in F[row].items():
            d = load_export(path, 'dev')
            mem = {f'{m}|{k}': q for (m, k), q in d.items() if m in methods}
            assert len(mem) == expected[row], (row, s, 'dev', len(mem))
            out[row][s] = mem
    return out


def gate_devsel(shadow, pop, fams, cells_csv, devsel_json):
    EXPORT, F, expected = math_families(shadow)
    rows = ['OSP-1', 'OSP-full', 'BELT-B', 'BELT-H', 'BELT-B-1', 'BELT-H-1']
    dev = devsel_families(F, expected, rows)
    nat_dev = load_export(EXPORT, 'dev')[(NATURAL, NATURAL)]
    got = devsel(dev, fams, nat_dev, pop.natural, pop, rows)
    for p in PRICES:
        for agg in ('micro', 'macro'):
            v = pop.slice_stats(pop.natural, float(p))['utility'][agg]
            got[f'Natural|{agg}|{p}'] = {'mean': v, 'per_seed': [v, v, v]}
    err, n = 0.0, 0
    for r in csv.DictReader(open(cells_csv)):
        if r['version'] != 'v2':
            continue
        g = got[f"{r['method']}|{r['aggregation']}|{r['report_lambda']}"]
        for x, y in zip(g['per_seed'], json.loads(r['per_seed'])):
            err = max(err, abs(x - y)); n += 1
    want = json.loads(Path(devsel_json).read_text())['cells']
    for p in PRICES:
        for agg in ('micro', 'macro'):
            w = want[f'OSP-full|{agg}|{p}']
            g = got[f'OSP-full|{agg}|{p}']
            assert [c.split('|', 1)[-1] for c in g['chosen']] == w['chosen'], (p, agg, g['chosen'], w['chosen'])
            for x, y in zip(g['per_seed'], w['per_seed']):
                err = max(err, abs(x - y)); n += 1
    return {'values_compared': n, 'max_abs_diff': err, 'ok': err <= 1e-12}, dev, nat_dev


def code_families(rows_root: Path, pool: str):
    fams = {r: {} for r in ('OSP-full', 'OSP-1', 'BELT-B', 'BELT-H', 'BELT-B-1', 'BELT-H-1')}
    dev = {r: {} for r in fams}
    natural, nat_dev, srcs = None, None, {}
    for s in SEEDS:
        path = rows_root / pool / 'rows' / f'{pool}_s{s}.json.gz'
        doc = json.loads(gzip.open(path, 'rt').read())
        assert doc['pool'] == pool and doc['seed'] == s and doc['columns'] == ['strict_accuracy', 'thinking_tokens', 'total_tokens']
        assert all(g.get('exact') for g in doc['meta']['gates'].values()), (path, doc['meta']['gates'])
        srcs[str(path)] = sha(path)
        lb = 'qvrs-lbdev' if s == 0 else f'qvrs-lbdev-s{s}'
        meth = {'OSP-full': 'OSP@dev160', 'OSP-1': 'OSP-single-threshold@dev160' + ('' if s == 0 else f'-s{s}'),
                'BELT-B': f'LB_QVRS@{lb}', 'BELT-H': f'LB_H@{lb}', 'BELT-B-1': f'LB_QVRS_1@olmo-thr-s{s}', 'BELT-H-1': f'LB_H_1@olmo-thr-s{s}'}
        by = {}
        for r in doc['rows']:
            by.setdefault(r['split'], {})[(r['method'], r['member'])] = r['q']
        nat = by['pool'][(NATURAL, NATURAL)]
        if natural is None:
            natural = nat
        assert nat == natural, (pool, s, 'natural differs across seeds')
        if (NATURAL, NATURAL) in by.get('dev', {}):
            nd = by['dev'][(NATURAL, NATURAL)]
            assert nat_dev is None or nd == nat_dev
            nat_dev = nd
        for row, m in meth.items():
            fams[row][s] = {f'{m}|{k}': q for (mm, k), q in by['pool'].items() if mm == m}
            dev[row][s] = {f'{m}|{k}': q for (mm, k), q in by['dev'].items() if mm == m}
    expected = {'OSP-full': 5, 'OSP-1': 13, 'BELT-B': 221, 'BELT-H': 221, 'BELT-B-1': 13, 'BELT-H-1': 13}
    for row, per in fams.items():
        for s, mem in per.items():
            assert len(mem) == expected[row] == len(dev[row][s]), (pool, row, s, len(mem), len(dev[row][s]))
            for q in mem.values():
                assert set(q) == set(natural)
    return fams, dev, natural, nat_dev, srcs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shadow', required=True, type=Path)
    ap.add_argument('--cells-csv', required=True, type=Path)
    ap.add_argument('--devsel', required=True, type=Path)
    ap.add_argument('--task9-result', required=True, type=Path)
    ap.add_argument('--rows', type=Path, help='members root (<pool>/rows/<pool>_s<k>.json.gz); omit for the gates only')
    ap.add_argument('--pools', nargs='*', default=[p for p, _, _ in POOLS])
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gm, mpop, mfams, mcells = gate_math(a.shadow)
    print('gate M', json.dumps(gm))
    gw = gate_wtl(mcells, a.task9_result)
    print('gate W', json.dumps(gw))
    ge, mdev, mnat_dev = gate_devsel(a.shadow, mpop, mfams, a.cells_csv, a.devsel)
    print('gate E', json.dumps(ge))
    res = {'schema': 'olcb-readings/v1', 'gates': {'M': gm, 'W': gw, 'E': ge},
           'inputs': {'shadow': str(a.shadow), 'cells_csv': sha(a.cells_csv), 'devsel': sha(a.devsel),
                      'task9_result': sha(a.task9_result),
                      'receipts': {f: sha(a.shadow / 'generated/framework' / f) for f in ('OLMO_OPERATING_RENDER_RECEIPT.json', 'OLMO_RENDER_RECEIPT.json')}},
           'pools': {}}
    if not (gm['ok'] and gw['ok'] and ge['ok']):
        (a.out / 'RESULTS.json').write_text(json.dumps(res, indent=1) + '\n')
        print('GATE FAILED; no code readings')
        return 3
    for pool in (a.pools if a.rows else []):
        fams, dev, natural, nat_dev, srcs = code_families(a.rows, pool)
        if nat_dev is None:
            nat_dev = mnat_dev
        assert nat_dev == mnat_dev, 'the code rows carry another dev natural'
        pop = Pop(natural, {q: pool for q in natural}, [pool])
        n_expect = dict((p, n) for p, _, n in POOLS)[pool]
        assert pop.n == n_expect, (pool, pop.n)
        rows = ['OSP-1', 'OSP-full', 'BELT-B-1', 'BELT-H-1', 'BELT-B', 'BELT-H']
        cells = Reader(pop, fams).all_cells(rows)
        for c in cells:
            if 'micro' in c:
                assert c['micro'] == c['macro'] and all(v['micro'] == v['macro'] for v in c['per_seed'].values())
        cmap = {(c['row'], c['kind'], c['level']): c for c in cells if 'per_seed' in c}
        ds = devsel(dev, fams, nat_dev, natural, pop, rows)
        for p in PRICES:
            v = pop.slice_stats(natural, float(p))['utility']['micro']
            ds[f'Natural|micro|{p}'] = {'mean': v, 'per_seed': [v, v, v]}
        res['pools'][pool] = {'n_questions': pop.n, 'natural': {'accuracy': 100 * pop.nat_micro[0], 'tokens': pop.nat_micro[1]},
                              'sources': srcs, 'cells': cells, 'wtl': wtl(cmap, 'micro'),
                              'devsel': {k: v for k, v in ds.items() if '|micro|' in k}}
        print(pool, 'natural acc %.2f tokens %.1f' % (100 * pop.nat_micro[0], pop.nat_micro[1]))
    (a.out / 'RESULTS.json').write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', a.out / 'RESULTS.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
