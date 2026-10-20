"""Prepare, fit, replay and collect learned sigmoid-linear stopping policies."""
import argparse
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

from rl_controller.iclr import families_learned_linear as lb
from rl_controller.iclr import families_qvrs_cpu as rules
from rl_controller.iclr import learned_linear_policy as lp
from rl_controller.scripts import iclr_qvrs_cpu_campaign as base

POOL = 'math_ood_report'
HEAD = 'qvrs-lbdev'
FAMILY_MODULES = ('rl_controller.iclr.families_learned_linear', 'rl_controller.iclr.families_qvrs_cpu')
EXPLICIT = ('XC_QONLY', 'XC_SONLY', 'XC_QSONLY', 'XC_QSADD', 'XC_QVR', 'XC_QVSAFE', 'XC_QVSMIX', 'XC_DEPTH')
BUNDLES = {
    'LB_QVRS': ('L1', 'LB_QV'),
    'LB_QV': ('L1',),
    'LB_QS': ('L1', 'LB_QV', 'XC_QSONLY', 'XC_QSADD'),
    'LB_DEPTH': ('L1', 'XC_DEPTH'),
    'LB_QV_D': ('L1', 'LB_QV'),
    'LB_QS_D': ('L1', 'LB_QS'),
    'LB_QVRS_D': ('L1', 'LB_QVRS'),
    **{name: ('L1',) for name in EXPLICIT},
}
SMOKE_BUNDLES = ('LB_QS', 'LB_DEPTH')
SMOKE_QUESTIONS = dict(fit=6, dev=6, select=3)
POPULATION = dict(val_questions=320, select_questions=951, select_chains=8400)
digest, atomic, completed, call_stage = base.digest, base.atomic, base.completed, base.call_stage


def task_matrix():
    return [dict(pool=POOL, family=family) for family in BUNDLES]


def candidate_counts():
    counts = {s.name: len(s.grid) for s in [*lb.specs(), *rules.specs()]}
    return {'L1': 13, **{name: counts[name] for f, rest in BUNDLES.items() for name in (f, *rest) if name != 'L1'}}


def registration():
    return dict(policy=lp.grids(), head_tag=HEAD, family_modules=list(FAMILY_MODULES),
                bundles={k: list(v) for k, v in BUNDLES.items()}, candidate_counts=candidate_counts(),
                learned_families=[s.to_dict() for s in lb.specs()],
                explicit_families=[s.to_dict() for s in rules.specs() if s.name in EXPLICIT])


def plan(family):
    others = BUNDLES[family]
    return [family, *others], [f'{family}@{HEAD} - {other}@{HEAD}' for other in others]


def validate_manifest(manifest, seeds=None):
    return base.validate_manifest(manifest, seeds=seeds)


def read_qkeys(path):
    with open(path) as f:
        return sorted({json.loads(line)['qkey'] for line in f if line.strip()})


def seed_entries(lock):
    return sorted(lock['manifest']['pools'][POOL], key=lambda e: e['seed'])


def check_population(counts, files, select_path, expected=None):
    with open(select_path) as f:
        chains = sum(1 for line in f if line.strip())
    if files.get(str(Path(select_path).resolve())) != digest(select_path):
        raise ValueError('evaluation dump changed while preparing')
    want = dict(POPULATION if expected is None else expected)
    got = dict(val_questions=counts['val'], select_questions=counts['select'], select_chains=chains)
    if got != want:
        raise ValueError(f'population {got} is not the registered {want}')
    return got


def prepare(args):
    from rl_controller.iclr import read_splits as rs
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()),
                                 seeds=getattr(args, 'expect_seeds', None))
    output = Path(args.out)
    if (output/'LOCK.json').exists():
        locked(args)
        print('Existing lock retained', flush=True)
        return
    files, counts, reader = base.bind_inputs(
        manifest, assets_lock=getattr(args, 'assets_lock', None),
        self_contained_reader=getattr(args, 'self_contained_reader', False),
        budget=getattr(args, 'budget', None))
    entries = sorted(manifest['pools'][POOL], key=lambda e: e['seed'])
    source = Path(entries[0]['run_dir'])/'val_scored.jsonl'
    if files.get(str(source.resolve())) != digest(source):
        raise ValueError('validation dump changed while preparing')
    expected = dict(POPULATION)
    for key, value in (('val_questions', getattr(args, 'expect_val_questions', None)),
                       ('select_questions', getattr(args, 'expect_select_questions', None)),
                       ('select_chains', getattr(args, 'expect_select_chains', None))):
        if value:
            expected[key] = int(value)
    population = check_population(counts[POOL], files,
                                 Path(entries[0]['run_dir'])/'select_scored.jsonl',
                                 expected=expected)
    fit, dev = lp.split_questions(read_qkeys(source))
    split = dict(salt=lp.SPLIT_SALT, rule='sha256(salt:qkey) ascending; first half policy-fit',
                 fit=fit, dev=dev, n_fit=len(fit), n_dev=len(dev),
                 fit_qkeys_sha256=rs.qkeys_sha256(fit), dev_qkeys_sha256=rs.qkeys_sha256(dev))
    atomic(output/'LOCK.json', dict(schema='learned-linear-lock/v1', manifest=manifest,
           manifest_sha256=digest(args.manifest), commit=base.revision(), files=files,
           grids=registration(), tasks=task_matrix(), question_counts=counts, population=population,
           assets_lock=(str(Path(args.assets_lock).resolve()) if getattr(args, 'assets_lock', None)
                        else None),
           self_contained_reader=bool(getattr(args, 'self_contained_reader', False)),
           budget=getattr(args, 'budget', None),
           feature_reader_sha256=reader, split=split, expected_pools=[POOL]))


def locked(args):
    lock = json.loads((Path(args.out)/'LOCK.json').read_text())
    if lock['manifest_sha256'] != digest(args.manifest) or lock['commit'] != base.revision() \
            or lock['grids'] != json.loads(json.dumps(registration())):
        raise ValueError('manifest, code revision or registered grids changed; use a new campaign output')
    return lock


def require_smoke(args):
    path = Path(args.out)/'SMOKE.json'
    record = json.loads(path.read_text()) if path.is_file() else {}
    if record.get('status') != 'passed' or record.get('lock_sha256') != digest(Path(args.out)/'LOCK.json'):
        raise ValueError('matching real-data smoke has not passed')


def parse_locked(path, locked_files, budget=None):
    from rl_controller.iclr.chains import load_chains
    from rl_controller.iclr.provenance import parse_and_record
    chains, record = parse_and_record(path, load_chains)
    if locked_files is not None and locked_files.get(str(Path(path).resolve())) != record['sha256']:
        raise ValueError(f'fit consumed unlocked bytes: {path}')
    for chain in chains:
        rules.validate_chain(chain, budget=budget)
    return chains, record['sha256']


def write_dump(path, chains, codes):
    tables = {lb.score_key(name): table.T.tolist() for name, table in codes.items()}
    tmp, offset = path.with_name(path.name + f'.{os.getpid()}.tmp'), 0
    with open(tmp, 'w') as out:
        for chain in chains:
            scores = []
            for i, score in enumerate(chain['scores']):
                if set(score) & set(tables):
                    raise ValueError(f"{chain['qkey']}: source scores already carry learned keys")
                scores.append({**score, **{key: rows[offset + i] for key, rows in tables.items()}})
            offset += len(chain['probes'])
            out.write(json.dumps(dict(chain, scores=scores), separators=(',', ':')) + '\n')
    if any(len(rows) != offset for rows in tables.values()):
        raise ValueError(f'{path}: rank codes do not align with probes')
    os.replace(tmp, path)


def plateau_summary(weights):
    models = [m for rows in weights['models'].values() for m in rows]
    return dict(rule=weights['grids']['plateau'], n_cells=len(models),
                fits_by_kappa={str(k): sum(f['plateau'] for m in models for f in m['fits'] if f['kappa'] == k)
                               for k in weights['grids']['kappas']},
                selected_plateau_cells={name: [m['lambda_index'] for m in rows if m['plateau']]
                                        for name, rows in weights['models'].items()})


def fit_seed(run_dir, out, *, fit, dev, locked_files=None, provenance=None, budget=None):
    from rl_controller.iclr import read_splits as rs
    run_dir, out = Path(run_dir), Path(out)
    started = time.monotonic()
    summary_path = run_dir/'summary.json'
    raw = summary_path.read_bytes()
    summary_sha, summary = hashlib.sha256(raw).hexdigest(), json.loads(raw)
    if locked_files is not None and locked_files.get(str(summary_path.resolve())) != summary_sha:
        raise ValueError(f'fit consumed unlocked bytes: {summary_path}')
    val, val_sha = parse_locked(run_dir/'val_scored.jsonl', locked_files, budget)
    select, select_sha = parse_locked(run_dir/'select_scored.jsonl', locked_files, budget)
    fit_set, dev_set, val_q = set(fit), set(dev), {c['qkey'] for c in val}
    if fit_set & dev_set or fit_set | dev_set != val_q:
        raise ValueError('policy split does not partition the source validation questions')
    if val_q & {c['qkey'] for c in select}:
        raise ValueError('source validation and evaluation questions overlap')
    fit_chains = [c for c in val if c['qkey'] in fit_set]
    dev_chains = [c for c in val if c['qkey'] in dev_set]
    weights = lp.fit_models(fit_chains, dev_chains)
    weights.update(seed=summary['config']['seed'], source_val_sha256=val_sha, split=dict(
        salt=lp.SPLIT_SALT, fit_qkeys_sha256=rs.qkeys_sha256(fit_set), dev_qkeys_sha256=rs.qkeys_sha256(dev_set),
        n_fit_questions=len(fit_set), n_dev_questions=len(dev_set),
        n_fit_chains=len(fit_chains), n_dev_chains=len(dev_chains)))
    atomic(out/'weights.json', weights)
    fit_seconds = time.monotonic() - started
    atomic(out/'diagnostics.json', dict(
        seed=weights['seed'], weights_sha256=digest(out/'weights.json'), fit_seconds=fit_seconds,
        plateau=plateau_summary(weights),
        stochastic_and_rank_counts={'fit': lp.diagnostics(weights, fit_chains),
                                    'dev': lp.diagnostics(weights, dev_chains),
                                    'ood': lp.diagnostics(weights, select)},
        note='OOD rows are exploratory reporting; nothing here is used for fitting or selection'))
    derived = out/'run'
    derived.mkdir(parents=True, exist_ok=True)
    write_dump(derived/'val_scored.jsonl', dev_chains, lp.deploy(weights, dev_chains))
    write_dump(derived/'select_scored.jsonl', select, lp.deploy(weights, select))
    summary.setdefault('provenance', {})['learned_linear_policy'] = dict(
        schema='iclr-learned-linear-derived/v1', derived_from=str(run_dir.resolve()),
        source_sha256=dict(summary=summary_sha, val_scored=val_sha, select_scored=select_sha),
        split=weights['split'], weights_sha256=digest(out/'weights.json'),
        diagnostics_sha256=digest(out/'diagnostics.json'),
        val_scored='policy-dev chains of the source validation dump, reader scores unchanged',
        select_scored=f'every {POOL} chain, reader scores unchanged',
        score_keys={lb.score_key(n): 'per probe: list over lambda index l of #{k: z_l >= tau_lk}'
                    for n in lp.FEATURE_SETS}, **(provenance or {}))
    atomic(derived/'summary.json', summary)
    print(json.dumps({'fitted_seed': weights['seed'], 'seconds': time.monotonic() - started}), flush=True)
    return [out/'weights.json', out/'diagnostics.json', derived/'summary.json',
            derived/'val_scored.jsonl', derived/'select_scored.jsonl']


def replay(run_dirs, out, family, jobs, identity, locked_files=None, n_seeds=None):
    families, pairs = plan(family)
    table = out/'questions.json'
    command = [sys.executable, '-u', '-m', 'rl_controller.scripts.iclr_ladder',
               '--runs', *map(str, run_dirs), '--out', str(out/'ladder.json'),
               '--md', str(out/'ladder.md'), '--no-calibration', '--bootstrap', '0',
               '--family-module', *FAMILY_MODULES, '--only-families', *families,
               '--export-question-tables', str(table), '--question-tables-scope', 'all',
               '--head-tag', HEAD, '--pool', POOL, '--overhead-accounting', 'none',
               '--answer-format', 'boxed']
    call_stage(command, out/'REPLAY_DONE.json', identity, [out/'ladder.json', out/'ladder.md', table])
    if locked_files is not None:
        base.check_consumed(table, locked_files, n_seeds=n_seeds)

    def axis(cost):
        command = [sys.executable, '-u', '-m', 'rl_controller.scripts.iclr_matched_contrasts',
                   '--question-tables', str(table), '--pool', POOL, '--basis', 'question',
                   '--cost', cost, '--jobs', str(max(1, jobs//2)),
                   '--out-dir', str(out/cost), '--name', 'paired']
        for pair in pairs:
            command += ['--contrast', pair]
        call_stage(command, out/f'{cost.upper()}_DONE.json', identity,
                   [out/cost/'paired.json', out/cost/'paired.md'])
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(axis, ('total', 'thinking')))


def write_subset(entry, target, keys):
    target.mkdir(parents=True, exist_ok=True)
    (target/'summary.json').write_bytes((Path(entry['run_dir'])/'summary.json').read_bytes())
    for split in ('val', 'select'):
        with open(Path(entry['run_dir'])/f'{split}_scored.jsonl') as f, \
                open(target/f'{split}_scored.jsonl', 'w') as out:
            out.writelines(line for line in f if line.strip() and json.loads(line)['qkey'] in keys[split])


def smoke(args):
    lock = locked(args)
    started, lock_sha = time.monotonic(), digest(Path(args.out)/'LOCK.json')
    entries = seed_entries(lock)
    base.verify_files(lock, entries)
    base.verify_sources(lock, entries)
    root = Path(args.out).resolve()/'smoke'
    if root.exists():
        root.rename(root.with_name(f'smoke.stale.{time.strftime("%Y%m%dT%H%M%S")}.{os.getpid()}'))
    fit, dev = lock['split']['fit'][:SMOKE_QUESTIONS['fit']], lock['split']['dev'][:SMOKE_QUESTIONS['dev']]
    if tuple(map(set, lp.split_questions(fit + dev))) != (set(fit), set(dev)):
        raise ValueError('split rule does not reproduce the locked assignment on the smoke subset')
    keys = dict(val=set(fit + dev),
                select=set(read_qkeys(Path(entries[0]['run_dir'])/'select_scored.jsonl')[:SMOKE_QUESTIONS['select']]))
    derived, fits, fit_outputs = [], {}, {}
    for entry in entries:
        name = f"s{entry['seed']}"
        write_subset(entry, root/'inputs'/name, keys)
        fit_started = time.monotonic()
        outputs = fit_seed(root/'inputs'/name, root/'fits'/name, fit=fit, dev=dev,
                           provenance=dict(smoke=True, lock_sha256=lock_sha),
                           budget=lock.get('budget'))
        fit_outputs.update({str(p.resolve()): digest(p) for p in outputs})
        fits[name] = time.monotonic() - fit_started
        derived.append(root/'fits'/name/'run')
    fit_outputs_sha = hashlib.sha256(json.dumps(fit_outputs, sort_keys=True).encode()).hexdigest()
    replays = {}
    for family in SMOKE_BUNDLES:
        out, replay_started = root/'replay'/family, time.monotonic()
        replay(derived, out, family, getattr(args, 'jobs', 4),
               dict(smoke=True, family=family, lock_sha256=lock_sha, fit_outputs_sha256=fit_outputs_sha),
               fit_outputs, n_seeds=len(entries))
        families, pairs = plan(family)
        if set(json.loads((out/'ladder.json').read_text())['rungs']) != set(families):
            raise ValueError(f'smoke family filter failed for {family}')
        for cost in ('total', 'thinking'):
            if len(json.loads((out/cost/'paired.json').read_text())['contrasts']) != len(pairs):
                raise ValueError(f'smoke {cost} contrasts incomplete for {family}')
        replays[family] = time.monotonic() - replay_started
    atomic(Path(args.out)/'SMOKE.json', dict(
        status='passed', lock_sha256=lock_sha, fit_outputs_sha256=fit_outputs_sha,
        questions={k: sorted(v) for k, v in keys.items()},
        scope=f'3 seeds, {SMOKE_QUESTIONS} questions, full fit grid and task commands {list(SMOKE_BUNDLES)}',
        fit_seconds=fits, replay_seconds=replays, elapsed_seconds=time.monotonic() - started))


def fit_dir(args, entry):
    return Path(args.out).resolve()/'fits'/f"s{entry['seed']}"


def fit_identity(lock_sha, entry):
    return dict(stage='fit', seed=entry['seed'], run_dir=entry['run_dir'], lock_sha256=lock_sha)


def fit(args):
    lock = locked(args)
    require_smoke(args)
    entries = seed_entries(lock)
    if args.task_id is None or not 0 <= args.task_id < len(entries):
        raise ValueError('invalid fit index')
    entry = entries[args.task_id]
    base.verify_files(lock, [entry])
    base.verify_sources(lock, [entry])
    out = fit_dir(args, entry)
    out.mkdir(parents=True, exist_ok=True)
    guard = open(out/'RUNNING.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_sha = digest(Path(args.out)/'LOCK.json')
    identity, receipt = fit_identity(lock_sha, entry), out/'FIT_DONE.json'
    if completed(receipt, identity):
        print(f'Resume: verified {receipt}', flush=True)
        return
    outputs = fit_seed(entry['run_dir'], out, fit=lock['split']['fit'], dev=lock['split']['dev'],
                       locked_files=lock['files'], budget=lock.get('budget'),
                       provenance=dict(lock_sha256=lock_sha, commit=lock['commit']))
    atomic(receipt, {'identity': identity, 'outputs': {str(p.resolve()): digest(p) for p in outputs}})


def fitted(args, lock):
    lock_sha, receipts, outputs = digest(Path(args.out)/'LOCK.json'), {}, {}
    for entry in seed_entries(lock):
        receipt = fit_dir(args, entry)/'FIT_DONE.json'
        if not completed(receipt, fit_identity(lock_sha, entry)):
            raise ValueError(f"fit for seed {entry['seed']} is missing or modified; rerun the fit stage")
        receipts[str(entry['seed'])] = digest(receipt)
        outputs.update(json.loads(receipt.read_text())['outputs'])
    return receipts, outputs


def task(args):
    lock = locked(args)
    require_smoke(args)
    if args.task_id is None or not 0 <= args.task_id < len(lock['tasks']):
        raise ValueError('invalid task id')
    task = lock['tasks'][args.task_id]
    receipts, fit_outputs = fitted(args, lock)
    out = Path(args.out).resolve()/'tasks'/f"{args.task_id:03d}_{task['pool']}_{task['family']}"
    out.mkdir(parents=True, exist_ok=True)
    guard = open(out/'RUNNING.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(task=task, lock_sha256=digest(Path(args.out)/'LOCK.json'), fit_receipts=receipts)
    runs = [fit_dir(args, e)/'run' for e in seed_entries(lock)]
    replay(runs, out, task['family'], args.jobs, identity, fit_outputs, n_seeds=len(runs))
    outputs = {}
    for receipt in ('REPLAY_DONE.json', 'TOTAL_DONE.json', 'THINKING_DONE.json'):
        outputs.update(json.loads((out/receipt).read_text())['outputs'])
    atomic(out/'DONE.json', {'identity': identity, 'outputs': outputs})


def collect(args):
    lock = locked(args)
    receipts, _ = fitted(args, lock)
    lock_sha, rows, missing = digest(Path(args.out)/'LOCK.json'), [], []
    for i, task in enumerate(lock['tasks']):
        directory = Path(args.out).resolve()/'tasks'/f"{i:03d}_{task['pool']}_{task['family']}"
        identity = dict(task=task, lock_sha256=lock_sha, fit_receipts=receipts)
        if not completed(directory/'DONE.json', identity):
            missing.append(i)
            continue
        families, pairs = plan(task['family'])
        rows.append(dict(task_id=i, **task, contrasts=pairs,
                         candidates={f: lock['grids']['candidate_counts'][f] for f in families},
                         total_json=str(directory/'total/paired.json'), thinking_json=str(directory/'thinking/paired.json'),
                         total=str(directory/'total/paired.md'), thinking=str(directory/'thinking/paired.md'),
                         questions=str(directory/'questions.json'),
                         outputs_sha256=json.loads((directory/'DONE.json').read_text())['outputs']))
    fits = {f"s{e['seed']}": dict(plateau=json.loads((fit_dir(args, e)/'diagnostics.json').read_text())['plateau'],
                                  weights=str(fit_dir(args, e)/'weights.json'),
                                  diagnostics=str(fit_dir(args, e)/'diagnostics.json'),
                                  run_dir=str(fit_dir(args, e)/'run'), receipt_sha256=receipts[str(e['seed'])])
            for e in seed_entries(lock)}
    atomic(Path(args.out)/'INDEX.json', dict(complete=not missing, missing_task_ids=missing,
           expected_pools=lock['expected_pools'], split={k: v for k, v in lock['split'].items()
                                                         if k not in ('fit', 'dev')},
           fits=fits, results=rows))
    if missing:
        raise ValueError(f'incomplete tasks: {missing}; resubmit these array indices')
    print(f'Collected {len(rows)} tasks; see INDEX.json', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('count', 'prepare', 'smoke', 'fit', 'task', 'collect'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--budget', type=int, default=None,
                        help="this line's registered generation cap (default: the 1.7B line's "
                             "12,000). A probe past the cap still fails; the cap is declared")
    parser.add_argument('--self-contained-reader', action='store_true',
                        help="this line's reader is trained per run rather than being an external "
                             "frozen head, so the extractor's identity is the run's own checkpoint "
                             "digest; registered for a single-seed line only")
    parser.add_argument('--assets-lock', default=None,
                        help='a registration whose chain banks are not the 1.7B line\'s; the check '
                             'is still made, against this file')
    parser.add_argument('--expect-seeds', type=int, nargs='+', default=None,
                        help='the seed set this line registered (default: 0 1 2). A single-seed '
                             'line declares it here rather than having the check relaxed')
    parser.add_argument('--expect-val-questions', type=int, default=None)
    parser.add_argument('--expect-select-questions', type=int, default=None)
    parser.add_argument('--expect-select-chains', type=int, default=None)
    args = parser.parse_args(argv)
    if args.stage == 'count':
        validate_manifest(json.loads(Path(args.manifest).read_text()),
                          seeds=getattr(args, 'expect_seeds', None))
        print(len(task_matrix()))
    else:
        if args.stage in ('smoke', 'fit'):
            import torch
            torch.set_num_threads(1)
        globals()[args.stage](args)


if __name__ == '__main__':
    main()
