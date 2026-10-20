import itertools
import math

from rl_controller.iclr import ladder as ld

BUDGET = 12000
MARGINS = tuple(round(-.3 + .05 * i, 2) for i in range(13))
THRESHOLDS = tuple(round(.02 + .08 * i, 2) for i in range(13))
QS_THRESHOLDS = (.5, .6, .7, .8, .9, .95, .99)
DEPTH_BETAS = (-.1, 0., .1)
GRIDS = {
    'QONLY': tuple((t,) for t in THRESHOLDS),
    'SONLY': tuple((t,) for t in THRESHOLDS),
    'QSONLY': tuple(itertools.product(QS_THRESHOLDS, repeat=2)),
    'QSADD': tuple((round(-.3 + .1 * i, 2),) for i in range(13)),
    'QVR': tuple(itertools.product(MARGINS, (0., .25, .5, 1.))),
    'QVSAFE': tuple(itertools.product(MARGINS, (.7, .9, .99), (.7, .9, .99))),
    'QVSMIX': tuple(itertools.product(MARGINS, (0., .5, 1.))),
    'QV': tuple((m,) for m in MARGINS),
    'DEPTH': tuple((i * 1000,) for i in range(13)),
}


def validate_chain(chain, budget=None):
    cap = BUDGET if budget is None else budget
    probes, scores = chain.get('probes', []), chain.get('scores', [])
    if len(probes) != len(scores):
        raise ValueError('probe/score length mismatch')
    previous = -1
    for probe, score in zip(probes, scores):
        length = probe.get('prefix_len')
        if not isinstance(length, int) or not previous <= length <= cap:
            raise ValueError('invalid/nonmonotone prefix_len or budget mismatch')
        previous = length
        for key in ('q', 'v', 'r', 'stability'):
            value = score.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'missing/invalid probability {key}')


def make_plan(kind, parameters, beta=0.):
    def plan(chain):
        for probe, s in zip(chain['probes'], chain['scores']):
            d = probe['prefix_len'] / BUDGET
            q, v, r, stability = (s[k] for k in ('q', 'v', 'r', 'stability'))
            p = parameters
            if kind == 'QONLY': value = q - p[0]
            elif kind == 'SONLY': value = stability - p[0]
            elif kind == 'QSONLY': value = min(q - p[0], stability - p[1])
            elif kind == 'QSADD': value = q + stability - 1 - p[0]
            elif kind == 'QV': value = q - v + p[0]
            elif kind == 'QVR': value = q - v + p[0] - p[1] * (1 - q) * r
            elif kind == 'QVSAFE': value = max(q - v + p[0], min(q-p[1], v-p[1], stability-p[2]))
            elif kind == 'QVSMIX': value = (1 - p[1] * stability) * (q - v) + p[0]
            elif kind == 'DEPTH': value = probe['prefix_len'] - p[0]
            else: raise ValueError(f'unknown policy {kind}')
            if value + beta * d >= 0:
                return probe
        return None
    return plan


def factory(*values, spec):
    name = spec.name.removeprefix('XC_')
    depth = name.endswith('_D')
    kind = name[:-2] if depth else name
    return make_plan(kind, values[:-1] if depth else values, values[-1] if depth else 0.)


def specs():
    result = []
    for kind, grid in GRIDS.items():
        for depth in (False, True):
            if (kind == 'QV' and not depth) or (kind == 'DEPTH' and depth):
                continue
            values = tuple((*p, b) for p in grid for b in DEPTH_BETAS) if depth else grid
            n = len(values[0])
            result.append(ld.FamilySpec(
                name='XC_' + kind + ('_D' if depth else ''), kind='custom',
                factory='qvrs_cpu_explicit', belief_key='q', requires=('v', 'r', 'stability'),
                grid=values, params=tuple(f'p{i}' for i in range(n)), formats=('{:.4f}',)*n,
                belief_source='2026-09-15 QVRS CPU registration; frozen QVRS reader'))
    return result


def register_defaults():
    ld.register_rule_factory('qvrs_cpu_explicit', factory)
    return [ld.register_family(s) for s in specs()]


DEFAULT_FAMILIES = tuple(register_defaults())
