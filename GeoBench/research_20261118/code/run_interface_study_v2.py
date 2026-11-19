"""Run GB-R3-A with fixed outer spatial evaluation and inner-only selection."""
from __future__ import annotations
import csv
import datetime
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
import sklearn
from study_core_v2 import METHODS, assign_folds, crossfit, predict, summarize_predictions, hole_rmse

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/GB-R3-B"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def load_inputs():
    with (BASE / "evidence/v2/eligible_contacts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("x", "y", "depth"):
            row[key] = float(row[key])
    return rows

def split_records(rows, mapping, dataset, outer, kind):
    return [{"dataset":dataset,"outer_fold":outer,"partition":kind,
             "borehole":hole,"assigned_fold":fold}
            for hole,fold in sorted(mapping.items())]

def run_fold(dataset, data, outer, fold):
    train = [r for r in data if outer[r["borehole"]] != fold]
    test = [r for r in data if outer[r["borehole"]] == fold]
    if {r["borehole"] for r in train} & {r["borehole"] for r in test}:
        raise ValueError("Outer borehole leakage")
    predictions, oof_rows, split_rows, selection = [], [], [], {}
    fits = {}
    # Selection and calibration only consume outer training observations.
    for kind in ("random", "spatial"):
        inner = assign_folds(train, 4, kind)
        split_rows.extend(split_records(train, inner, dataset, fold, kind))
        fitted = crossfit(train, inner)
        fits[kind] = fitted
        selected = min(METHODS, key=lambda m: fitted[m]["rmse"])
        selection[kind] = {"selected":selected, "candidates":{}}
        for method in METHODS:
            info = fitted[method]
            selection[kind]["candidates"][method] = {k:v for k,v in info.items() if k not in ("predictions","scales")}
            for row, value, scale in zip(train, info["predictions"], info["scales"]):
                supported = bool(np.isfinite(value))
                oof_rows.append({"dataset":dataset,"outer_fold":fold,"validation":kind,
                                 "method":method,"target_id":row["target_id"],
                                 "borehole":row["borehole"],"inner_fold":inner[row["borehole"]],
                                 "truth":row["depth"],"prediction":value,"scale":scale,
                                 "supported":supported})
    # Select both validation designs on exactly the same held-out target population.
    common = np.isfinite(fits["random"][METHODS[0]]["predictions"]) & np.isfinite(fits["spatial"][METHODS[0]]["predictions"])
    common_rows = [r for r, keep in zip(train, common) if keep]
    for kind in ("random", "spatial"):
        scores = {}
        for method in METHODS:
            score = hole_rmse(common_rows, fits[kind][method]["predictions"][common])
            scores[method] = score
            selection[kind]["candidates"][method]["common_support_rmse"] = score
            selection[kind]["candidates"][method]["common_support_targets"] = len(common_rows)
        selection[kind]["selected"] = min(METHODS, key=lambda m: scores[m])
    # Refit each fixed estimator once, then compare calibration on identical outputs.
    for method in METHODS:
        values, scales, distances = predict(train, test, method)
        for kind in ("random", "spatial"):
            q = selection[kind]["candidates"][method]["radius"]
            for row, value, scale, distance in zip(test, values, scales, distances):
                supported = bool(np.isfinite(value))
                predictions.append({"dataset":dataset,"outer_fold":fold,"method":method,
                                    "calibration":kind,"selected":selection[kind]["selected"]==method,
                                    "target_id":row["target_id"],"borehole":row["borehole"],
                                    "unit":row["unit"],"truth":row["depth"],"prediction":value,
                                    "scale":scale,"nearest_unit_distance_m":distance,"radius":q,
                                    "lower":max(0.,value-q*scale) if supported else float("nan"),
                                    "upper":value+q*scale if supported else float("nan"),
                                    "supported":supported})
    masks = defaultdict(set)
    for row in predictions:
        if row["supported"]:
            masks[(row["method"],row["calibration"])].add(row["target_id"])
    if len({frozenset(v) for v in masks.values()}) != 1:
        raise ValueError("Outer support differs between estimators")
    return predictions, oof_rows, split_rows, selection

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    tracked = [BASE / "protocol_B.md", BASE / "evidence/v2/eligible_contacts.csv",
               BASE / "sources/rvg_v0.1.0_Data.zip", BASE / "sources/rvg_record.json",
               BASE.parents[1] / "核心论文一/reproducibility/audit_source.py",
               BASE.parents[1] / "source_file/钻孔及分布资料/航空港分层信息表909.xlsx",
               *[BASE / "code" / name for name in ("data_sources_v2.py", "study_core_v2.py", "run_interface_study_v2.py")]]
    hashes = {str(p.resolve().relative_to(BASE.parents[1])):digest(p) for p in tracked}
    manifest_path = OUT / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["hashes"] != hashes:
            raise ValueError("Run inputs changed: create a new run ID instead of overwriting")
    manifest = {"status":"running","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "hashes":hashes,"environment":{"python":platform.python_version(),"numpy":np.__version__,
                "scipy":scipy.__version__,"sklearn":sklearn.__version__},
                "exploratory":True,"protocol":"GB-R3-B"}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    all_predictions, all_splits, all_selection = [], [], {}
    # Each fold has a checkpoint, so interrupted work resumes without refitting finished folds.
    for dataset in sorted({r["dataset"] for r in inputs}):
        data = [r for r in inputs if r["dataset"] == dataset]
        with (BASE / "results/GB-R3-A/split_assignments.csv").open() as stream:
            outer = {r["borehole"]:int(r["assigned_fold"]) for r in csv.DictReader(stream) if r["dataset"]==dataset and r["partition"]=="outer"}
        all_splits.extend(split_records(data, outer, dataset, -1, "outer"))
        for fold in range(5):
            start=time.monotonic()
            checkpoint=OUT / (dataset + "_fold" + str(fold) + ".json")
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                predictions,splits,selection=saved["predictions"],saved["splits"],saved["selection"]
            elif dataset == "zhengzhou":
                original = BASE / "results/GB-R3-A" / checkpoint.name
                saved = json.loads(original.read_text())
                predictions,splits,selection = saved["predictions"],saved["splits"],saved["selection"]
                source = BASE / "results/GB-R3-A" / f"{dataset}_fold{fold}_oof.csv"
                (OUT/source.name).write_bytes(source.read_bytes())
                checkpoint.write_text(json.dumps(saved)+"\n")
            else:
                print(f"START {dataset} fold={fold}",flush=True)
                predictions,oof,splits,selection=run_fold(dataset,data,outer,fold)
                write_csv(OUT / f"{dataset}_fold{fold}_oof.csv",oof)
                checkpoint.write_text(json.dumps({"predictions":predictions,"splits":splits,"selection":selection})+"\n")
            all_predictions.extend(predictions)
            all_splits.extend(splits)
            all_selection[f"{dataset}_{fold}"]=selection
            print(f"DONE {dataset} fold={fold} seconds={time.monotonic()-start:.1f} "
                  f"selected={selection['random']['selected']}/{selection['spatial']['selected']}",flush=True)
    write_csv(OUT / "predictions.csv",all_predictions)
    write_csv(OUT / "split_assignments.csv",all_splits)
    (OUT / "selection.json").write_text(json.dumps(all_selection,indent=2)+"\n")
    summaries={}
    for dataset in sorted({r["dataset"] for r in inputs}):
        for kind in ("random","spatial"):
            for method in (*METHODS,"selected"):
                rows=[r for r in all_predictions if r["dataset"]==dataset and r["calibration"]==kind
                      and (r["selected"] if method=="selected" else r["method"]==method)]
                summaries[f"{dataset}/{kind}/{method}"]=summarize_predictions(rows)
    (OUT / "summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    manifest.update(status="complete",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    prediction_sha256=digest(OUT/"predictions.csv"),records=len(all_predictions))
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summaries.items() if k.endswith("/selected")},indent=2),flush=True)

if __name__ == "__main__":
    main()

