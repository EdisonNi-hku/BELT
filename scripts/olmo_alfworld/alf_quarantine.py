from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

GENERATED = ("seg", "full", "fa:", "natural", "sc:")


def block_of(path: str) -> str:
    tag = os.path.basename(path)[len("nodes."):-len(".jsonl")] if os.path.basename(path) != "nodes.jsonl" else "pilot"
    return tag.split("-w")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    total = 0
    for lane in sorted(glob.glob(f"{a.tree}/lane*/")):
        src = defaultdict(lambda: defaultdict(set))
        for f in sorted(glob.glob(f"{lane}/nodes*.jsonl")):
            blk = block_of(f)
            for line in open(f):
                if not line.endswith("\n"):
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("v") == 2:
                    continue
                for part in r:
                    if part.startswith(GENERATED):
                        src[r["key"]][part.split("#")[0]].add(blk)
        keys = sorted(k for k, parts in src.items() if any(len(b) > 1 for b in parts.values()))
        total += len(keys)
        rec = {"keys": keys, "why": "parts written by more than one member block", "blocks":
               sorted({b for k in keys for bs in src[k].values() for b in bs}),
               "made": datetime.datetime.now().isoformat(timespec="seconds")}
        print(json.dumps({"lane": lane, "quarantined": len(keys), "blocks": rec["blocks"]}))
        if not a.dry_run:
            q = Path(lane) / "quarantine.json"
            if q.exists():
                old = json.loads(q.read_text())["keys"]
                rec["keys"] = sorted(set(old) | set(keys))
            q.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"total": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
