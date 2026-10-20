from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d53lib as L


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_patched.py --producer P [--needs ...] (-m MOD | --script F) -- ARGS")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--producer", required=True)
    ap.add_argument("--needs", nargs="*", default=["chain_terms_total"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-m", dest="module")
    g.add_argument("--script")
    a = ap.parse_args(argv[:cut])
    rec = L.setup(a.producer, needs=tuple(a.needs))
    print("[run_patched] " + json.dumps({"producer": rec["producer"], "fallback_cost": rec["fallback_cost"]["charges"],
                                         "d53_code": rec["d53_code"]["commit"]}), flush=True)
    target = a.module or a.script
    sys.argv = [target] + argv[cut + 1:]
    if a.module:
        runpy.run_module(a.module, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(a.script, run_name="__main__")


if __name__ == "__main__":
    main()
