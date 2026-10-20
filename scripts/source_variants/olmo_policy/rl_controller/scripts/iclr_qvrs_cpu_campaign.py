"""Prepare, replay and collect CPU belief-policy fits."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rl_controller.iclr import families_qvrs_cpu as rules

POOLS = ('select', 'math_t1_report', 'math_t2_report', 'math_ood_report', 'ood2_math_report')
EXPECTED_POOLS = ('math_ood_report',)
HEAD = 'qvrs-cpu'
SMOKE_FAMILIES = ('XC_QSONLY', 'XC_QSONLY_D')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def validate_manifest(manifest, seeds=None):
    want = {0, 1, 2} if seeds is None else set(seeds)
    pools = manifest.get('pools', {})
    if not pools or set(pools) - set(POOLS):
        raise ValueError('pools must be a nonempty subset of the five already-opened pools')
    if set(pools) != set(EXPECTED_POOLS):
        raise ValueError('pilot requires only math_ood_report; expansion needs a new registration')
    for pool, entries in pools.items():
        if len(entries) != len(want) or any(not isinstance(e, dict) for e in entries):
            raise ValueError(f'{pool}: need {len(want)} entries with seed/run_dir/checkpoint')
        if {e.get('seed') for e in entries} != want:
            raise ValueError(f'{pool}: require seeds {sorted(want)} exactly')
        for entry in entries:
            for key in ('run_dir', 'checkpoint'):
                if not isinstance(entry.get(key), str) or not Path(entry[key]).is_absolute():
                    raise ValueError(f'{pool}: {key} must be an absolute path')
    return manifest


def task_matrix(manifest):
    return [dict(pool=pool, family=s.name)
            for pool in POOLS if pool in manifest['pools'] for s in rules.specs()]


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def registration():
    return [s.to_dict() for s in rules.specs()]


DEFAULT_ASSETS_LOCK = 'docs/superpowers/contracts/2026-09-08-two-belief-assets.lock.json.txt'


def expected_banks(lock_path=None):
    path = (Path(lock_path) if lock_path
            else Path(__file__).resolve().parents[2]/DEFAULT_ASSETS_LOCK)
    return json.loads(path.read_text())['chain_banks_euler']


def source_run_dir(summary, entry):
    recorded = summary.get('provenance', {}).get('source_run') or summary['config'].get('source_run')
    if not recorded:
        return None
    if (Path(recorded)/'summary.json').is_file():
        return Path(recorded)
    checkpoint = Path(entry['checkpoint'])
    if checkpoint.name != 's1_best.pt' or checkpoint.parent.name != Path(recorded).name:
        raise ValueError(f"{entry['run_dir']}: source_run {recorded} is unreadable here and the "
                         'manifest checkpoint is not s1_best.pt in a directory of that name')
    return checkpoint.parent


def feature_reader(prov, fallback=None):
    head = prov.get('head_checkpoint') or {}
    if 'loaded' in head:
        if head.get('match') is not True or head.get('override') is not False:
            raise ValueError('scored with an unmatched or overridden feature reader')
        return (head.get('loaded') or {}).get('weights_sha256')
    declared = (head.get('weights') or {}).get('sha256')
    return declared if declared is not None else fallback


def check_consumed(table, files, n_seeds=None):
    expected = 3 if n_seeds is None else int(n_seeds)
    sources = json.loads(Path(table).read_text())['sources']
    if len(sources) != expected or any(set(splits) != {'val', 'select'} for splits in sources.values()):
        raise ValueError(f'{table}: expected val/select sources for {expected} reader seed(s), '
                         f'got {len(sources)}')
    for seed, splits in sources.items():
        for split, record in splits.items():
            key = str(Path(record['path']).resolve(strict=True))
            if files.get(key) != record['sha256']:
                raise ValueError(f'replay consumed unlocked bytes: {key} (seed {seed}, {split})')


def prepare(args):
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    output = Path(args.out)
    if (output / 'LOCK.json').exists():
        locked(args)
        print('Existing lock retained', flush=True)
        return
    files, counts, reader = bind_inputs(manifest)
    atomic(output/'LOCK.json', dict(schema='qvrs-cpu-lock/v2', manifest=manifest,
           manifest_sha256=digest(args.manifest), commit=revision(), files=files,
           grids=registration(), tasks=task_matrix(manifest), question_counts=counts,
           feature_reader_sha256=reader,
           expected_pools=list(EXPECTED_POOLS),
           incomplete_pool_coverage=sorted(set(EXPECTED_POOLS)-set(manifest['pools']))))


def bind_inputs(manifest, assets_lock=None, self_contained_reader=False, budget=None):
    from rl_controller.scripts import iclr_ladder as ladder
    files, signatures, checkpoints, counts, readers = {}, {}, {}, {}, set()
    for pool in POOLS:
        if pool not in manifest['pools']:
            continue
        runs = []
        for entry in sorted(manifest['pools'][pool], key=lambda e: e['seed']):
            path, seed = Path(entry['run_dir']), entry['seed']
            for file in (path/'summary.json', path/'val_scored.jsonl', path/'select_scored.jsonl',
                         Path(entry['checkpoint'])):
                resolved = str(file.resolve(strict=True))
                if resolved not in files:
                    files[resolved] = digest(resolved)
            run = ladder.load_run(path)
            if run['seed'] != seed or set(run['targets']) != {'q', 'v', 'r', 'stability'}:
                raise ValueError(f'{path}: require matching seed and exact QVRS training targets')
            identities = ladder.chain_identity_by_split(run, ('val', 'select'))
            prov = run['summary'].get('provenance', {})
            read_bank = identities['select'].get('chain_bank_sha256')
            val_bank = identities['val'].get('chain_bank_sha256')
            ck = files[str(Path(entry['checkpoint']).resolve())]
            source = source_run_dir(run['summary'], entry)
            original_training = (path/'s1_best.pt').is_file() and source is None
            if original_training:
                nested = prov.get('select_chains', {})
                nested_bank = nested.get('sha256') if isinstance(nested, dict) else None
                if read_bank and nested_bank and read_bank != nested_bank:
                    raise ValueError(f'{path}: conflicting evaluation bank declarations')
                read_bank = read_bank or nested_bank
                val_bank = val_bank or ladder.declared_fit_bank(prov)
            if source is not None:
                nested = prov.get('chains', {})
                nested_bank = nested.get('sha256') if isinstance(nested, dict) else None
                if read_bank and nested_bank and read_bank != nested_bank:
                    raise ValueError(f'{path}: conflicting evaluation bank declarations')
                read_bank = read_bank or nested_bank
                source_summary = str((source/'summary.json').resolve(strict=True))
                files[source_summary] = digest(source_summary)
                source_data = json.loads(Path(source_summary).read_text())
                if source_data['config'].get('seed') != seed or \
                        set(ladder._targets(source_data['config'])) != {'q', 'v', 'r', 'stability'}:
                    raise ValueError(f'{source}: source run seed/targets differ')
                if (source/'s1_best.pt').is_file() and digest(source/'s1_best.pt') != ck:
                    raise ValueError(f'{source}: source run checkpoint differs from the manifest checkpoint')
                if digest(source/'val_scored.jsonl') != run['dumps']['val']['sha256']:
                    raise ValueError(f'{path}: validation dump is not the source run validation dump')
                if feature_reader(prov, fallback=ck if self_contained_reader else None) != \
                        feature_reader(source_data.get('provenance', {}),
                                       fallback=ck if self_contained_reader else None):
                    raise ValueError(f'{path}: report scored with a different feature reader than its training run')
                val_bank = val_bank or ladder.declared_fit_bank(source_data.get('provenance', {}))
            banks = expected_banks(assets_lock)
            read_key = 'math_union_select' if pool == 'select' else pool
            if read_bank != banks[read_key]['sha256'] or val_bank != banks['math_union_fit']['sha256']:
                raise ValueError(f'{path}: evaluation pool/validation provenance does not match registered banks')
            readers.add(feature_reader(prov, fallback=ck if self_contained_reader else None))
            if None in readers or len(readers) != 1:
                raise ValueError(f'{path}: feature reader undeclared or differs across runs: {sorted(map(str, readers))}')
            recorded = (prov.get('s1_checkpoint') or {}).get('sha256')
            if recorded:
                if recorded != ck:
                    raise ValueError(f'{path}: scored checkpoint digest differs')
            elif (path/'s1_best.pt').is_file():
                if digest(path/'s1_best.pt') != ck:
                    raise ValueError(f'{path}: original training checkpoint differs')
            else:
                raise ValueError(f'{path}: no s1_checkpoint provenance or original s1_best.pt')
            if seed in checkpoints and checkpoints[seed] != ck:
                raise ValueError('different reader checkpoints across pools')
            checkpoints[seed] = ck
            val_sha = run['dumps']['val']['sha256']
            if seed in signatures and signatures[seed] != val_sha:
                raise ValueError('validation dump differs across pools; do not select on report')
            signatures[seed] = val_sha
            for split in ('val', 'select'):
                ids = [(c['qkey'], c['rep']) for c in run[split]]
                if len(ids) != len(set(ids)):
                    raise ValueError(f'{path}: duplicate {split} chain identity')
                for chain in run[split]:
                    rules.validate_chain(chain, budget=budget)
                if not any(c['probes'] for c in run[split]):
                    raise ValueError(f'{path}: no eligible probes')
            if {c['qkey'] for c in run['val']} & {c['qkey'] for c in run['select']}:
                raise ValueError(f'{path}: validation/evaluation question overlap')
            runs.append(run)
        ladder.check_runs(runs)
        for split in ('val', 'select'):
            canonical = []
            for run in runs:
                stripped = [{k: v for k, v in c.items() if k != 'scores'} for c in run[split]]
                stripped.sort(key=lambda c: (c['qkey'], c['rep']))
                canonical.append(hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest())
            if len(set(canonical)) != 1:
                raise ValueError(f'{pool}: {split} bank/labels differ across reader seeds')
        counts[pool] = {split: len({c['qkey'] for c in runs[0][split]}) for split in ('val', 'select')}
        del runs
        print(json.dumps({'prepared_pool': pool, 'questions': counts[pool]}), flush=True)
    return files, counts, readers.pop() if readers else None


def locked(args):
    lock = json.loads((Path(args.out)/'LOCK.json').read_text())
    if lock['manifest_sha256'] != digest(args.manifest) or lock['commit'] != revision() \
            or lock['grids'] != json.loads(json.dumps(registration())):
        raise ValueError('manifest, code revision or registered grids changed; use a new campaign output')
    return lock


def verify_files(lock, entries):
    for entry in entries:
        for file in (Path(entry['run_dir'])/'summary.json', Path(entry['run_dir'])/'val_scored.jsonl',
                     Path(entry['run_dir'])/'select_scored.jsonl', Path(entry['checkpoint'])):
            key = str(file.resolve(strict=True))
            if lock['files'].get(key) != digest(key):
                raise ValueError(f'locked input changed: {key}')


def verify_sources(lock, entries):
    for entry in entries:
        summary = json.loads((Path(entry['run_dir'])/'summary.json').read_text())
        source = source_run_dir(summary, entry)
        if source is not None:
            key = str((source/'summary.json').resolve(strict=True))
            if digest(key) != lock['files'].get(key):
                raise ValueError('source training provenance changed')


def write_subset(entry, target, n_questions=3):
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(entry['run_dir'])/'summary.json', target/'summary.json')
    kept = {}
    for split in ('val', 'select'):
        source = Path(entry['run_dir'])/f'{split}_scored.jsonl'
        with open(source) as f:
            keys = set(sorted({json.loads(line)['qkey'] for line in f if line.strip()})[:n_questions])
        with open(source) as f, open(target/f'{split}_scored.jsonl', 'w') as out:
            out.writelines(line for line in f if line.strip() and json.loads(line)['qkey'] in keys)
        kept[split] = sorted(keys)
    return kept


def smoke(args):
    lock = locked(args)
    started, pools = time.monotonic(), {}
    lock_sha = digest(Path(args.out)/'LOCK.json')
    for pool in dict.fromkeys(t['pool'] for t in lock['tasks']):
        entries = sorted(lock['manifest']['pools'][pool], key=lambda e: e['seed'])
        verify_files(lock, entries)
        verify_sources(lock, entries)
        root = Path(args.out).resolve()/'smoke'/pool
        kept = [write_subset(e, root/f"s{e['seed']}") for e in entries]
        if any(k != kept[0] for k in kept):
            raise ValueError(f'{pool}: reader seeds disagree on the smoke questions')
        pool_started = time.monotonic()
        for family in SMOKE_FAMILIES:
            out = root/'replay'/family
            replay([str(root/f"s{e['seed']}") for e in entries], out, pool, family,
                   getattr(args, 'jobs', 4), dict(smoke=pool, family=family, lock_sha256=lock_sha))
            families, pairs = plan(family)
            if set(json.loads((out/'ladder.json').read_text())['rungs']) != set(families):
                raise ValueError(f'{pool}: smoke family filter failed for {family}')
            for cost in ('total', 'thinking'):
                if len(json.loads((out/cost/'paired.json').read_text())['contrasts']) != len(pairs):
                    raise ValueError(f'{pool}: smoke {cost} contrasts incomplete for {family}')
        pools[pool] = dict(questions=kept[0], elapsed_seconds=time.monotonic()-pool_started)
    atomic(Path(args.out)/'SMOKE.json', {'status': 'passed', 'lock_sha256': lock_sha,
           'scope': f'every locked pool, 3 questions per split, 3 seeds, full task commands {list(SMOKE_FAMILIES)}',
           'pools': pools, 'elapsed_seconds': time.monotonic()-started})


def completed(receipt, identity):
    if not receipt.is_file():
        return False
    data = json.loads(receipt.read_text())
    if data.get('identity') != identity:
        raise ValueError(f'stale receipt {receipt}')
    return all(Path(p).is_file() and digest(p) == sha for p, sha in data['outputs'].items())


def call_stage(command, receipt, identity, outputs):
    if completed(receipt, identity):
        print(f'Resume: verified {receipt}', flush=True)
        return
    print(json.dumps({'command': command}), flush=True)
    subprocess.run(command, check=True)
    atomic(receipt, {'identity': identity, 'outputs': {str(p.resolve()): digest(p) for p in outputs}})


def plan(family):
    base = family[:-2] if family.endswith('_D') and family != 'XC_QV_D' else None
    controls = ['XC_QONLY', 'XC_SONLY'] if family in ('XC_QSONLY', 'XC_QSADD') else []
    families = ['L1', family] + ([base] if base else []) + controls
    pairs = [f'{family}@{HEAD} - {other}@{HEAD}' for other in ['L1'] + ([base] if base else []) + controls]
    return families, pairs


def replay(run_dirs, out, pool, family, jobs, identity, locked_files=None):
    families, pairs = plan(family)
    table = out/'questions.json'
    command = [sys.executable, '-u', '-m', 'rl_controller.scripts.iclr_ladder',
               '--runs', *run_dirs, '--out', str(out/'ladder.json'),
               '--md', str(out/'ladder.md'), '--no-calibration', '--bootstrap', '0',
               '--family-module', 'rl_controller.iclr.families_qvrs_cpu',
               '--only-families', *families, '--export-question-tables', str(table),
               '--question-tables-scope', 'all', '--head-tag', HEAD, '--pool', pool,
               '--overhead-accounting', 'none', '--answer-format', 'boxed']
    call_stage(command, out/'REPLAY_DONE.json', identity, [out/'ladder.json', out/'ladder.md', table])
    if locked_files is not None:
        check_consumed(table, locked_files)

    def axis(cost):
        target = out/cost
        command = [sys.executable, '-u', '-m', 'rl_controller.scripts.iclr_matched_contrasts',
                   '--question-tables', str(table), '--pool', pool, '--basis', 'question',
                   '--cost', cost, '--jobs', str(max(1, jobs//2)),
                   '--out-dir', str(target), '--name', 'paired']
        for pair in pairs:
            command += ['--contrast', pair]
        call_stage(command, out/f'{cost.upper()}_DONE.json', identity,
                   [target/'paired.json', target/'paired.md'])
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(axis, ('total', 'thinking')))


def task(args):
    lock = locked(args)
    smoke_record = json.loads((Path(args.out)/'SMOKE.json').read_text())
    if smoke_record.get('status') != 'passed' or smoke_record['lock_sha256'] != digest(Path(args.out)/'LOCK.json'):
        raise ValueError('matching real-data smoke has not passed')
    if args.task_id is None or not 0 <= args.task_id < len(lock['tasks']):
        raise ValueError('invalid task id')
    task = lock['tasks'][args.task_id]
    pool, family = task['pool'], task['family']
    entries = lock['manifest']['pools'][pool]
    verify_files(lock, entries)
    verify_sources(lock, entries)
    out = Path(args.out).resolve()/'tasks'/f'{args.task_id:03d}_{pool}_{family}'
    out.mkdir(parents=True, exist_ok=True)
    task_guard = open(out/'RUNNING.lock', 'a')
    fcntl.flock(task_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(task=task, lock_sha256=digest(Path(args.out)/'LOCK.json'))
    replay([e['run_dir'] for e in entries], out, pool, family, args.jobs, identity, lock['files'])
    outputs = {}
    for receipt in ('REPLAY_DONE.json', 'TOTAL_DONE.json', 'THINKING_DONE.json'):
        outputs.update(json.loads((out/receipt).read_text())['outputs'])
    atomic(out/'DONE.json', {'identity': identity, 'outputs': outputs})


def collect(args):
    lock = locked(args)
    rows, missing = [], []
    for i, task in enumerate(lock['tasks']):
        directory = Path(args.out).resolve()/'tasks'/f"{i:03d}_{task['pool']}_{task['family']}"
        identity = dict(task=task, lock_sha256=digest(Path(args.out)/'LOCK.json'))
        if not completed(directory/'DONE.json', identity):
            missing.append(i)
        else:
            receipt = json.loads((directory/'DONE.json').read_text())['outputs']
            rows.append(dict(task_id=i, **task, total=str(directory/'total/paired.md'),
                             thinking=str(directory/'thinking/paired.md'),
                             total_json=str(directory/'total/paired.json'),
                             thinking_json=str(directory/'thinking/paired.json'),
                             questions=str(directory/'questions.json'),
                             outputs_sha256=receipt))
    atomic(Path(args.out)/'INDEX.json', dict(complete=not missing, missing_task_ids=missing,
           expected_pools=lock['expected_pools'],
           incomplete_pool_coverage=lock['incomplete_pool_coverage'], results=rows))
    if missing:
        raise ValueError(f'incomplete tasks: {missing}; resubmit these array indices')
    print(f'Collected {len(rows)} tasks; see INDEX.json', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('count', 'prepare', 'smoke', 'task', 'collect'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args(argv)
    if args.stage == 'count':
        manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
        print(len(task_matrix(manifest)))
    else:
        globals()[args.stage](args)


if __name__ == '__main__':
    main()
