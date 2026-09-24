#!/usr/bin/env python3
"""Phase F: write the pre-evaluation lock.

Everything the locked evaluation will use is hashed here, before ``G_check`` is
touched.  After this file exists, no configuration change is permitted; the
evaluation module reads its checkpoints and its predictor penalty from this record
and refuses to overwrite its own output, so a second, adjusted run cannot quietly
replace the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..candidate_effect_distillation_v1 import contract as ced
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.open_vocab_dual_encoder_v1.data import sha256
from . import contract as fc


CODE_PACKAGES = ("final_clean_model_v1", "candidate_effect_distillation_v1", "utility_gt_v2",
                 "action_conditioned_utility_v1")


def _code_digest(root: Path) -> dict:
    """Hash every source file the locked evaluation depends on.

    The authoring checkout has uncommitted work, so a commit hash alone does not
    identify the code that produced this lock.  This does, and it does not require
    changing the repository state in the middle of a freeze.
    """
    digest = hashlib.sha256()
    files = []
    for package in CODE_PACKAGES:
        for path in sorted((root / "gene_open_inverse" / package).glob("*.py")):
            payload = path.read_bytes()
            digest.update(path.name.encode("utf-8"))
            digest.update(hashlib.sha256(payload).digest())
            files.append({"path": f"gene_open_inverse/{package}/{path.name}",
                          "sha256": hashlib.sha256(payload).hexdigest()})
    return {"packages": list(CODE_PACKAGES), "combined_sha256": digest.hexdigest(), "files": files}


def _git_commit(repository: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as error:  # the remote mirror is not a git checkout
        return f"UNAVAILABLE: {error.__class__.__name__}"


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("the lock refuses to overwrite an existing lock record")
    config = AuditConfig.load(Path(args.config))
    selection = json.loads(Path(args.selection).read_text())
    training = json.loads((Path(args.checkpoints) / "train_manifest.json").read_text())
    distillation = json.loads(
        (Path(args.distillation) / "candidate_effect_distillation.json").read_text())

    record = {
        "schema": "VCDESIGN_FINAL_LOCK_V1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit or _git_commit(Path(args.repository)),
        "git_working_tree_uncommitted_files": args.uncommitted,
        "code_digest": _code_digest(Path(args.repository)),
        "git_commit_note": "the authoring checkout has uncommitted work, so code_digest, not the "
                           "commit hash, identifies the code that produced this lock",
        "method_of_record": fc.FINAL_METHOD,
        "no_config_change_allowed_after_G_check_access": fc.NO_CONFIG_CHANGE_ALLOWED_AFTER_G_CHECK_ACCESS,
        "evaluation_split_name": fc.EVALUATION_SPLIT_NAME,
        "evaluation_split_must_not_be_called": fc.EVALUATION_SPLIT_MUST_NOT_BE_CALLED,
        "G_check_never_used_to_select": {
            "method": True, "effect_predictor": True, "fusion": True, "checkpoints": True,
            "query_gate": True, "candidate_feature_contract": True,
            "note": "every selection in this programme was made on G_fit or G_select; earlier "
                    "phases of the wider programme did read G_check, which is why this is a locked "
                    "final evaluation split and not an untouched blind test",
        },
        "query_gate": {"rule": "QUERY_ELIGIBILITY_V1", "fdr": 0.05,
                       "applied": "unchanged, at production A/B depth, to whichever role is evaluated"},
        "candidate_feature_contract": {
            "primary_modalities": list(fc.PRIMARY_MODALITIES),
            "TEXT": fc.TEXT_STATUS, "TEXT_reason": fc.TEXT_REASON,
            "all_missing_rule": fc.ALL_MISSING_RULE,
            "all_missing_contribution": fc.ALL_MISSING_CONTRIBUTION,
            "identical_in_both_arms": True,
        },
        "effect_predictor": {
            "basis_rank": ced.BASIS_RANK, "basis_centered": ced.BASIS_CENTERED,
            "basis_seed": ced.BASIS_SEED, "response_target": ced.CONSENSUS,
            "ridge_penalty": float(distillation["predictor"]["cross_validation"]["selected_penalty"]),
            "ridge_criterion": ced.RIDGE_CRITERION, "ridge_folds": ced.RIDGE_FOLDS,
            "fit_population": ced.RIDGE_FIT_POPULATION,
            "selected_entirely_on": "G_fit",
            "artifact": {"path": str(Path(args.distillation) / "predicted_effect_G_select.npy"),
                         "sha256": sha256(Path(args.distillation) / "predicted_effect_G_select.npy")},
        },
        "architecture": {
            "class": "gene_open_inverse.final_clean_model_v1.model.CleanAttributionModel",
            "difference_from_incumbent": "no learnable unknown token; all-missing candidates "
                                         "contribute exactly zero",
            "parameters": training["parameters"], "modality_dimensions": training["modality_dimensions"],
        },
        "training_recipe": training["recipe"],
        "fusion": {"form": fc.FINAL_FUSION, "beta": fc.FINAL_BETA,
                   "selected_before_G_check": fc.FINAL_FUSION_SELECTED_BEFORE_G_CHECK,
                   "reason": fc.FINAL_FUSION_SELECTION_REASON},
        "selected_checkpoints": {
            arm: {"epoch": entry["epoch"], "checkpoint": entry["checkpoint"],
                  "sha256": entry["sha256"], "G_select_MBRU_median": entry["G_select_MBRU_median"],
                  "selected_on": "G_select"}
            for arm, entry in selection["selected"].items()},
        "evaluation": {
            "primary_metrics": ["MBRU", "Mean@10", "Mean@20", "Mean@50",
                                "HighValueCount@10", "HighValueCount@20", "HighValueCount@50"],
            "guardrail_metrics": ["J RU@10", "J RU@20", "J RU@50",
                                  "fraction J>0 @10", "fraction J>0 @20"],
            "statistic": "paired identity-cluster bootstrap, 10000 replicates, seed 20260916",
            "primary_comparison": "CLEAN_EFFECT minus CLEAN_BASE",
            "comparator_note": "legacy M_SET is context only and is not the paired causal comparator",
            "rows": ["RANDOM", "STRICT_LEGAL_CANDIDATE_PRIOR", "EFFECT_BRANCH_ONLY", "LEGACY_M_SET",
                     "CLEAN_BASE", "CLEAN_EFFECT", "MEASURED_RESPONSE_ORACLE"],
            "output_schema": "one row per policy with direction, guardrail and all-missing prefix "
                             "blocks, plus a paired block per comparison",
        },
        "assets": config.provenance(),
        "runtime": {"python": platform.python_version()},
    }
    payload = json.dumps(record, indent=2, sort_keys=True)
    record["lock_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase F: pre-G_check lock")
    for name in ("config", "selection", "checkpoints", "distillation", "repository", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--git-commit", dest="git_commit", default=None,
                        help="commit of the authoring checkout, since the remote mirror is not a git tree")
    parser.add_argument("--uncommitted", type=int, default=-1)
    record = run(parser.parse_args())
    print(f"lock written, sha256 {record['lock_sha256']}")
    for arm, entry in record["selected_checkpoints"].items():
        print(f"  {arm}: epoch {entry['epoch']} sha256 {entry['sha256'][:16]}")
    print(f"  fusion {record['fusion']['form']} beta={record['fusion']['beta']}")
    print(f"  modalities {record['candidate_feature_contract']['primary_modalities']}, "
          f"TEXT {record['candidate_feature_contract']['TEXT']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
