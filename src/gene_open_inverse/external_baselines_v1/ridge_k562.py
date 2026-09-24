#!/usr/bin/env python3
"""GEARS's information-matched internal twin: the frozen A1 recipe on the K562-only atlas, masked
with exactly GEARS's fold-union masks (runs in the ``boom`` env).

The SEEN fit must equal the evidence run's saved K562-only spec to 1e-10 before anything is kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evidence_v1 import run as xr
from ..four_context_v1 import contract as fc
from ..four_context_v1 import harness as hn
from ..open_vocab_generalization_v1 import masking as mk
from ..ppm_v1 import run as pr
from . import contract as cc
from .gears_fit import fold_union

SPEC = {"name": "K562", "contexts": ("K562",), "matched": False, "modality": "both", "rank": 256}


def main() -> int:
    parser = argparse.ArgumentParser(description="RIDGE_UNIT on the K562 atlas under GEARS's masks")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--packs", default=str(Path(cc.FOUR_CONTEXT_RUN) / "packs"))
    args = parser.parse_args()
    output = Path(args.run_dir) / "ridge_k562"
    output.mkdir(parents=True, exist_ok=False)
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    unions = {f: fold_union(cc.C018_RUN, cc.FOUR_CONTEXT_RUN, f) for f in range(cc.MASK_FOLDS)}
    record = {"mask_rule": cc.GEARS_MASK_RULE, "fold_union_sizes": {f: len(u) for f, u in unions.items()}}
    for held in cc.PRIMARY_CONTEXTS:
        setup = pr.held_setup(packs, held)
        pack = setup["pack"]
        eligible, folds = setup["eligible"], setup["folds"]
        seen, info = xr.fit_spec(packs, SPEC, np.asarray([], dtype=str), 0, pack)
        evidence = np.load(Path(cc.EVIDENCE_RUN) / "fits" / held / cc.SEEN_POOL / "effects.npz", allow_pickle=False)["K562"]
        gate = float(np.abs(seen - evidence).max())
        if gate > 1e-10:
            raise RuntimeError(f"{held}: SEEN K562 atlas differs from the evidence run by {gate}")
        parts = []
        for fold in range(cc.MASK_FOLDS):
            effect, part_info = xr.fit_spec(packs, SPEC, np.asarray(unions[fold], dtype=str), 0, pack)
            inside = set(pack.ids[eligible & (folds == fold)].astype(str).tolist())
            if not inside <= set(unions[fold]):
                raise RuntimeError(f"{held} fold {fold}: a masked identity is outside the union")
            parts.append(effect)
        masked = mk._crossfit(seen, parts, eligible, folds)
        np.savez(output / f"effects_{held}.npz", SEEN=seen, MASKED=masked)
        record[held] = {"seen_vs_evidence_max_abs": gate, "seen_rows": info["rows"], "seen_penalty": info["penalty"]}
    (output / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
