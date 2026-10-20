from rl_controller.iclr import ladder as ld

FEATURE_SETS = ('DEPTH', 'QV', 'QV_D', 'QS', 'QS_D', 'QVRS', 'QVRS_D')
N_LAMBDAS = 17
N_LEVELS = 13
FACTORY = 'learned_linear_rank'


def score_key(feature_set):
    return f'lb_{feature_set}'


def make_plan(key, lam_index, level):
    def plan(chain):
        for probe, score in zip(chain['probes'], chain['scores']):
            if score[key][lam_index] > level:
                return probe
        return None
    return plan


def factory(lam_index, level, *, spec):
    return make_plan(spec.belief_key, lam_index, level)


def specs():
    return [ld.FamilySpec(
        name=f'LB_{name}', kind='custom', factory=FACTORY, belief_key=score_key(name),
        grid=tuple((l, k) for l in range(N_LAMBDAS) for k in range(N_LEVELS)),
        params=('l', 'k'), formats=('{:02d}', '{:02d}'),
        belief_source='2026-09-15 lane B learned sigmoid-linear stop policy; lambda index l, '
                      'policy-fit logit quantile level k/12; frozen QVRS reader')
        for name in FEATURE_SETS]


def register_defaults():
    ld.register_rule_factory(FACTORY, factory)
    return [ld.register_family(s) for s in specs()]


DEFAULT_FAMILIES = tuple(register_defaults())
