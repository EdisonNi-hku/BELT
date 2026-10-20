from __future__ import annotations


def _nat_osp(chain: dict) -> int:
    if chain.get("outcome", "natural") != "natural":
        return 0
    return int(chain.get("natural_fallback_tokens", 0))


def _nat_ta(chain, *, answer_format, force_natural_answer, encode, programs):
    if chain["outcome"] != "natural":
        return 0
    if answer_format == "code":
        key = (chain["qkey"], int(chain["rep"]), "natural", None)
        try:
            record = programs[key] if programs is not None else None
        except KeyError:
            record = None
        if record is None:
            raise AccountingError(f"natural code chain {key[:2]} has no program sidecar")
        if int(record["prefix_len"]) != int(chain["thinking_tokens"]):
            raise AccountingError(f"program sidecar prefix mismatch for {key[:2]}")
        return int(record["tokens"])
    if answer_format != "boxed":
        raise AccountingError(f"unsupported answer_format {answer_format!r}")
    if not force_natural_answer:
        return int(chain.get("natural_fallback_tokens", 0))
    if chain.get("answer") is None:
        raise AccountingError("forced natural answer is missing from an answered chain")
    if encode is None:
        raise AccountingError("forced boxed natural accounting requires the pinned tokenizer")
    return len(encode(f"{chain['answer']}}}"))


def _probe_chain(tokens=7, outcome="natural"):
    return {"qkey": "_probe", "rep": 0, "gold": 1, "outcome": outcome, "answer": 1, "correct": True,
            "blocks": 1, "thinking_tokens": 10, "block_ends": [], "query_points": [], "probes": [],
            "natural_fallback_tokens": tokens}


def _read(osp, ta, chain):
    return (int(osp.natural_answer_tokens_archived_math(chain)),
            int(ta._natural_answer_tokens(chain, answer_format="boxed", force_natural_answer=False,
                                          encode=None, programs=None)))


def install(osp=None, ta=None) -> dict:
    if osp is None:
        from rl_controller.iclr import osp_policy as osp
    if ta is None:
        from rl_controller.iclr import token_accounting as ta
    before = _read(osp, ta, _probe_chain())
    if before == (7, 7):
        mode = "native"
    elif before == (0, 0):
        for fn, new in ((osp.natural_answer_tokens_archived_math, _nat_osp),
                        (ta._natural_answer_tokens, _nat_ta)):
            if fn.__code__.co_freevars or fn.__code__.co_argcount != new.__code__.co_argcount \
                    or fn.__code__.co_kwonlyargcount != new.__code__.co_kwonlyargcount:
                raise SystemExit(f"{fn.__module__}.{fn.__name__}: signature differs from the "
                                 "aa3910bb body; refusing to patch")
            fn.__code__ = new.__code__
        mode = "patched"
    else:
        raise SystemExit(f"producer natural-cost functions return {before} on a fallback chain; "
                         "neither the legacy constant 0 nor the aa3910bb field read")
    after = _read(osp, ta, _probe_chain())
    capped = _read(osp, ta, _probe_chain(outcome="length"))
    plain = _probe_chain()
    del plain["natural_fallback_tokens"]
    none = _read(osp, ta, plain)
    if after != (7, 7) or capped != (0, 0) or none != (0, 0):
        raise SystemExit(f"fallback cost self-check failed: {after} {capped} {none}")
    try:
        terms = osp.chain_terms(_probe_chain(), cost="total")
    except TypeError:
        terms = osp.chain_terms(_probe_chain())
    charges = {"deployment_cost": ta.deployment_cost(_probe_chain(), answer_format="boxed")["generated_tokens"] == 17,
               "chain_terms_total": terms.l_total_nat == 17.0}
    rc = getattr(osp, "replay_chain", None)
    if rc is not None:
        c = dict(_probe_chain(), scores=[])
        charges["replay_chain"] = rc(c)["e_answer_tokens"] == 7.0
    return {"mode": mode, "osp_module": osp.__file__, "ta_module": ta.__file__, "charges": charges,
            "self_check": "fallback chain charged 7 by both functions; capped and field-less rows 0"}
