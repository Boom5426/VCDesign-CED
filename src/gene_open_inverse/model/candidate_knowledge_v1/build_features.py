#!/usr/bin/env python3
"""Build static candidate features without reading any perturbation response or relevance."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import AutoModel, AutoTokenizer

from .common import (array_sha256, dump_json_exclusive, load_json, resolve_config,
                     save_npz_exclusive, sha256_file, sha256_tree, standardise_present)


def log(message, started=[time.time()]):
    print("[%7.1fs] %s" % (time.time() - started[0], message), flush=True)


def coverage(present, positions):
    p = np.asarray(present, bool)[np.asarray(positions, np.int64)]
    return {"present": int(p.sum()), "total": int(len(p)), "fraction": float(p.mean())}


def load_axes(cfg):
    split = load_json(cfg["paths"]["split"])
    assert split["schema"] == "GOAL_CONDITIONAL_POP_V1_INNER_SPLIT"
    z = np.load(cfg["paths"]["string"], allow_pickle=False)
    universe = z["genes"].astype(str)
    z.close()
    uindex = {g: i for i, g in enumerate(universe)}
    pos = {role: np.asarray([uindex[g] for g in split[role]], np.int64)
           for role in ("G_fit", "G_select", "G_check")}
    assert len(set(split["G_fit"]) | set(split["G_select"]) | set(split["G_check"])) == 14220
    return split, universe, pos


def build_string(cfg, universe, fit_pos):
    z = np.load(cfg["paths"]["string"], allow_pickle=False)
    assert np.array_equal(z["genes"].astype(str), universe)
    raw = z["Z"].astype(np.float32)
    present = (z["degree"].astype(np.int64) > 0) & np.isfinite(raw).all(1) & (np.linalg.norm(raw, axis=1) > 0)
    z.close()
    value, mu, sd, keep, rows = standardise_present(raw, present, fit_pos, cfg["feature"]["sd_floor"])
    return value, present, {"raw_dim": 512, "output_dim": 512, "fit_present": len(rows),
                            "nondegenerate_dimensions": int(keep.sum()),
                            "mu_sha256": array_sha256(mu), "sd_sha256": array_sha256(sd)}


def build_sequence(cfg, universe, fit_pos):
    z = np.load(cfg["paths"]["sequence"], allow_pickle=False)
    assert np.array_equal(z["genes"].astype(str), universe)
    raw = z["E"].astype(np.float32)
    present = z["have"].astype(bool) & np.isfinite(raw).all(1)
    z.close()
    raw_std, raw_mu, raw_sd, raw_keep, rows = standardise_present(
        raw, present, fit_pos, cfg["feature"]["sd_floor"])
    ncomp = int(cfg["feature"]["sequence_components"])
    pca = PCA(n_components=ncomp, svd_solver=cfg["feature"]["sequence_pca_solver"],
              iterated_power=int(cfg["feature"]["sequence_pca_iterated_power"]),
              random_state=int(cfg["feature"]["random_state"]))
    pca.fit(raw_std[rows])
    projected = pca.transform(raw_std).astype(np.float32)
    value, mu, sd, keep, rows2 = standardise_present(
        projected, present, fit_pos, cfg["feature"]["sd_floor"])
    assert np.array_equal(rows, rows2)
    meta = {"raw_dim": int(raw.shape[1]), "output_dim": int(value.shape[1]),
            "fit_present": len(rows), "raw_nondegenerate_dimensions": int(raw_keep.sum()),
            "nondegenerate_dimensions": int(keep.sum()),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "raw_mu_sha256": array_sha256(raw_mu), "raw_sd_sha256": array_sha256(raw_sd),
            "pca_mean_sha256": array_sha256(pca.mean_),
            "pca_components_sha256": array_sha256(pca.components_),
            "final_mu_sha256": array_sha256(mu), "final_sd_sha256": array_sha256(sd)}
    del raw, raw_std, projected
    return value, present, meta


def reactome_membership(path, universe):
    index = {g: i for i, g in enumerate(universe)}
    entries = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".gmt")]
        assert len(names) == 1, names
        with archive.open(names[0]) as binary:
            for raw in binary:
                fields = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                members = sorted({index[g] for g in fields[2:] if g in index})
                entries.append((fields[0], fields[1], members))
    entries.sort(key=lambda item: (item[1], item[0]))
    rows, cols = [], []
    for col, (_, _, members) in enumerate(entries):
        rows.extend(members)
        cols.extend([col] * len(members))
    data = np.ones(len(rows), np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(universe), len(entries)), dtype=np.float32)
    return matrix, entries


def build_function(cfg, universe, fit_pos):
    raw, entries = reactome_membership(cfg["paths"]["reactome_gmt_zip"], universe)
    fit_counts = np.asarray(raw[fit_pos].sum(axis=0)).ravel()
    keep_path = fit_counts >= int(cfg["feature"]["reactome_min_gfit_members"])
    retained = raw[:, keep_path].tocsr()
    retained_counts = fit_counts[keep_path]
    retained = retained @ sp.diags(1.0 / np.sqrt(retained_counts), format="csr")
    present = np.asarray(retained.getnnz(axis=1) > 0).ravel()
    rows = fit_pos[present[fit_pos]]
    ncomp = int(cfg["feature"]["function_components"])
    assert retained.shape[1] > ncomp and len(rows) > ncomp
    svd = TruncatedSVD(n_components=ncomp, n_iter=int(cfg["feature"]["function_svd_iterations"]),
                       random_state=int(cfg["feature"]["random_state"]))
    svd.fit(retained[rows])
    projected = svd.transform(retained).astype(np.float32)
    value, mu, sd, keep, rows2 = standardise_present(
        projected, present, fit_pos, cfg["feature"]["sd_floor"])
    assert np.array_equal(rows, rows2)
    stable_ids = np.asarray([entry[1] for entry, flag in zip(entries, keep_path) if flag])
    meta = {"raw_pathways": int(raw.shape[1]), "retained_pathways": int(retained.shape[1]),
            "output_dim": int(value.shape[1]), "fit_present": int(len(rows)),
            "nondegenerate_dimensions": int(keep.sum()),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "retained_stable_ids_sha256": array_sha256(stable_ids),
            "svd_components_sha256": array_sha256(svd.components_),
            "final_mu_sha256": array_sha256(mu), "final_sd_sha256": array_sha256(sd)}
    return value, present, meta


def ncbi_descriptions(path, universe):
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["#tax_id"] != "9606":
                continue
            fields = [row.get("description", ""), row.get("Full_name_from_nomenclature_authority", ""),
                      row.get("Other_designations", "")]
            fields = [x.strip() for x in fields if x and x.strip() != "-"]
            text = ". ".join(dict.fromkeys(fields))
            aliases = [] if row.get("Synonyms", "-") == "-" else row["Synonyms"].split("|")
            records.append({"symbol": row["Symbol"], "aliases": aliases, "text": text,
                            "gene_id": row["GeneID"], "modified": row.get("Modification_date", "")})
    direct = {row["symbol"]: row for row in records}
    alias = defaultdict(list)
    for row in records:
        for name in row["aliases"]:
            alias[name].append(row)
    texts, mapping, gene_ids, dates = [], [], [], []
    for gene in universe:
        if gene in direct:
            row, how = direct[gene], "exact_symbol"
        elif len(alias.get(gene, [])) == 1:
            row, how = alias[gene][0], "unique_alias"
        else:
            row, how = None, "missing_or_ambiguous"
        if row is None or not row["text"]:
            texts.append(""); mapping.append("missing"); gene_ids.append(""); dates.append("")
        else:
            texts.append(row["text"]); mapping.append(how); gene_ids.append(row["gene_id"]); dates.append(row["modified"])
    return texts, np.asarray(mapping), np.asarray(gene_ids), np.asarray(dates)


def encode_text(cfg, texts, present, device):
    model_path = cfg["paths"]["text_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    out = np.zeros((len(texts), int(model.config.hidden_size)), np.float32)
    indices = np.flatnonzero(present)
    batch_size = int(cfg["feature"]["text_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            token = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                              max_length=int(cfg["feature"]["text_max_tokens"]), return_tensors="pt")
            token = {key: value.to(device) for key, value in token.items()}
            hidden = model(**token).last_hidden_state
            mask = token["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[rows] = pooled.float().cpu().numpy()
            if start % (batch_size * 20) == 0:
                log("TEXT encoded %d/%d" % (min(start + batch_size, len(indices)), len(indices)))
    return out


def build_text(cfg, universe, fit_pos, device):
    texts, mapping, gene_ids, dates = ncbi_descriptions(cfg["paths"]["ncbi_gene_info"], universe)
    present = np.asarray([bool(text) for text in texts], bool)
    raw = encode_text(cfg, texts, present, device)
    value, mu, sd, keep, rows = standardise_present(raw, present, fit_pos, cfg["feature"]["sd_floor"])
    tree_hash, _ = sha256_tree(cfg["paths"]["text_encoder"])
    meta = {"raw_dim": int(raw.shape[1]), "output_dim": int(value.shape[1]),
            "fit_present": int(len(rows)), "nondegenerate_dimensions": int(keep.sum()),
            "mapping_counts": {name: int((mapping == name).sum()) for name in np.unique(mapping)},
            "mapped_gene_ids_sha256": array_sha256(gene_ids),
            "source_modification_dates_sha256": array_sha256(dates),
            "encoder_tree_sha256": tree_hash,
            "final_mu_sha256": array_sha256(mu), "final_sd_sha256": array_sha256(sd)}
    return value, present, meta


def main():
    if not __debug__:
        raise RuntimeError("assertions are part of the protocol")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = resolve_config(args.config)
    split, universe, role_pos = load_axes(cfg)
    fit_pos = role_pos["G_fit"]
    inputs = {name: {"path": cfg["paths"][name], "sha256": sha256_file(cfg["paths"][name])}
              for name in ("string", "sequence", "reactome_gmt_zip", "ncbi_gene_info")}
    inputs["text_encoder"] = {"path": cfg["paths"]["text_encoder"],
                              "tree_sha256": sha256_tree(cfg["paths"]["text_encoder"])[0]}

    values, presence, details = {}, {}, {}
    values["STRING"], presence["STRING"], details["STRING"] = build_string(cfg, universe, fit_pos)
    log("STRING complete")
    values["SEQUENCE"], presence["SEQUENCE"], details["SEQUENCE"] = build_sequence(cfg, universe, fit_pos)
    log("SEQUENCE complete")
    values["FUNCTION"], presence["FUNCTION"], details["FUNCTION"] = build_function(cfg, universe, fit_pos)
    log("FUNCTION complete")
    values["TEXT"], presence["TEXT"], details["TEXT"] = build_text(cfg, universe, fit_pos, args.device)
    log("TEXT complete")

    meta = {"schema": "CANDIDATE_KNOWLEDGE_V1_FEATURE_MANIFEST",
            "status": "STATIC_FEATURES_COMPLETE_NO_RESPONSE_READ",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "governance": {"perturbation_response_read": False, "relevance_read": False,
                           "G_select_or_G_check_used_to_fit_transform": False,
                           "official_DEV_TEST_response_or_relevance_opened": False},
            "sha256": {"config": sha256_file(args.config), "design": sha256_file(args.design),
                       "split": sha256_file(cfg["paths"]["split"]), "inputs": inputs},
            "feature_details": details, "coverage": {}}
    arrays = {"universe": universe, "fit_pos": fit_pos}
    for name in ("STRING", "SEQUENCE", "FUNCTION", "TEXT"):
        arrays[name] = values[name]
        arrays[name + "_present"] = presence[name]
        meta["coverage"][name] = {role: coverage(presence[name], positions)
                                  for role, positions in role_pos.items()}
        meta["coverage"][name]["universe"] = coverage(presence[name], np.arange(len(universe)))
        meta["feature_details"][name]["array_sha256"] = array_sha256(values[name])
    arrays["meta"] = np.asarray(json.dumps(meta, sort_keys=True))
    digest = save_npz_exclusive(args.out, **arrays)
    meta["asset"] = {"path": str(Path(args.out).resolve()), "sha256": digest}
    dump_json_exclusive(args.manifest, meta)
    log("FEATURES_DONE %s" % digest)


if __name__ == "__main__":
    main()
