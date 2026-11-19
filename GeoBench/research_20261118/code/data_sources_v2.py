"""Traceable conditional basal-contact inputs; raw files remain unchanged."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
ARCHIVE = BASE / "sources/rvg_v0.1.0_Data.zip"
EXPECTED = "4fe7f97f340c3ecd471cf59f65de17faa8f1cd9c0ba3b9304ce9e079a85fa391"
TOL = 0.011

def rvg_metadata(records):
    output = {}
    # Dutch lithology endpoints are elevations; convert the terminus to depth.
    for row in records:
        values = {"x": float(row["x"]), "y": float(row["y"]),
                  "surface": float(row["mv"]),
                  "hole_depth": float(row["mv"]) - float(row["end"])}
        if not all(math.isfinite(v) for v in values.values()) or values["hole_depth"] <= 0:
            raise ValueError("Invalid borehole metadata: " + row["nr"])
        old = output.get(row["nr"])
        if old is not None and any(abs(old[k] - v) > TOL for k, v in values.items()):
            raise ValueError("Inconsistent borehole metadata: " + row["nr"])
        output[row["nr"]] = values
    return output

def merge_segments(rows):
    merged = []
    # Only contiguous records with the same unit represent one logged segment.
    for row in sorted(rows, key=lambda r: (r["top"], r["depth"], r["source_rows"])):
        if merged and merged[-1]["unit"] == row["unit"] and abs(merged[-1]["depth"] - row["top"]) <= TOL:
            merged[-1]["depth"] = row["depth"]
            merged[-1]["source_rows"] += "|" + row["source_rows"]
            merged[-1]["original_units"] += "|" + row["original_units"]
        else:
            merged.append(dict(row))
    return merged

def annotate_hole(rows, metadata):
    merged = merge_segments(rows)
    counts = Counter(r["unit"] for r in merged)
    # A basal target requires an observed adjacent next segment, not just a number.
    for index, row in enumerate(merged):
        flags = []
        if row["unit"] == "NN":
            flags.append("unknown_unit")
        if counts[row["unit"]] > 1:
            flags.append("disconnected_repeated_unit")
        if row["top"] < -TOL or row["depth"] <= row["top"]:
            flags.append("invalid_interval")
        if row["depth"] > metadata["hole_depth"] + TOL:
            flags.append("beyond_hole_depth")
        if index == len(merged) - 1:
            flags.append("terminal")
        else:
            if merged[index + 1]["unit"] == "NN":
                flags.append("unknown_next_unit")
            delta = merged[index + 1]["top"] - row["depth"]
            if delta > TOL:
                flags.append("gap_after")
            if delta < -TOL:
                flags.append("overlap_after")
        if index and row["top"] < merged[index - 1]["depth"] - TOL:
            flags.append("overlap_before")
        row["eligible"] = not flags
        row["reason"] = "|".join(flags)
        row["target_id"] = "rvg:" + row["borehole"] + ":" + row["source_rows"]
    return merged

def rvg_contacts(records, metadata, unit_map=None):
    unit_map = unit_map or {}
    grouped = defaultdict(list)
    conflicts = {}
    for index, row in enumerate(records, 2):
        hole = row["NR"]
        meta = metadata[hole]
        top, bottom = float(row["TOP"]), float(row["BASIS"])
        if not math.isfinite(top) or not math.isfinite(bottom) or not row["STRAT"]:
            raise ValueError("Invalid stratigraphy row: " + str(index))
        # Check duplicate metadata fields without using them as target features.
        for field, key in (("X", "x"), ("Y", "y"), ("MV", "surface")):
            if field in row and abs(float(row[field]) - meta[key]) > TOL:
                conflicts.setdefault(hole, {})[field] = {"stratigraphy": float(row[field]), "lithology": meta[key]}
        grouped[hole].append({"dataset": "rvg", "borehole": hole,
                             "unit": unit_map.get(row["STRAT"], row["STRAT"]), "original_units": row["STRAT"], "top": top, "depth": bottom,
                             "source_rows": str(row.get("_row", index)), **meta})
    output = []
    for hole in sorted(grouped):
        annotated = annotate_hole(grouped[hole], metadata[hole])
        if hole in conflicts:
            for row in annotated:
                row["eligible"] = False
                row["reason"] = "|".join(filter(None, [row["reason"], "metadata_conflict"]))
        output.extend(annotated)
    audit = {"raw_segments": sum(len(v) for v in grouped.values()),
             "merged_segments": len(output), "metadata_conflicts": conflicts,
             "reasons": dict(Counter(flag for r in output for flag in r["reason"].split("|") if flag))}
    return output, audit

def load_rvg():
    payload = ARCHIVE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED:
        raise ValueError("Unexpected RVG archive version")
    # Read only the two authoritative data members; ignore OS resource forks.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        def rows(suffix):
            names = [n for n in archive.namelist() if n.startswith("Data/") and n.endswith(suffix)]
            if len(names) != 1:
                raise ValueError("Ambiguous archive member")
            return list(csv.DictReader(io.StringIO(archive.read(names[0]).decode("utf-8-sig"))))
        lith = rows("basisdata_gef_bovennaaronder_public_domain_RVG.csv")
        strat = rows("DGMplus_extractie_DGM_v02r2_en_H3OdeKempen_NL_public_domain_RVG.csv")
        mapping_rows = rows("strat_class_map.csv")
    mapping = {r["key"]: r["value"] for r in mapping_rows}
    contacts, audit = rvg_contacts(strat, rvg_metadata(lith), mapping)
    audit["group_mapping"] = mapping
    audit.update({"archive_sha256": EXPECTED, "lithology_rows": len(lith),
                  "interpretation": "Expert lithostratigraphic interpretation, not new field truth",
                  "code_mapping": "Author-supplied strat_class_map.csv, all groups retained; no outcome-driven rare-group removal"})
    return contacts, audit

def load_zhengzhou():
    sys.path.insert(0, str(ROOT / "核心论文一/reproducibility"))
    from audit_source import load_observations, WORKBOOK, sha256
    observations, excluded, terminal = load_observations()
    output = []
    # Retain the existing contact policy and its exact row provenance.
    for row in observations:
        flags = ([row["qc_flags"]] if row["qc_flags"] else []) + (["terminal"] if row["terminal"] else [])
        output.append({"dataset": "zhengzhou", "borehole": row["borehole"], "unit": row["unit"],
                       "top": row["record_start"], "depth": row["depth"], "original_units": row["unit"],
                       "source_rows": str(row["source_row"]), "x": row["x"], "y": row["y"],
                       "surface": row["surface"], "hole_depth": row["hole_depth"],
                       "eligible": row["geometry_eligible"], "reason": "|".join(flags),
                       "target_id": "zhengzhou:" + str(row["source_row"])})
    return output, {"workbook_sha256": sha256(WORKBOOK), "raw_records": len(observations),
                    "terminal": terminal, "excluded_source_rows": excluded,
                    "interpretation": "Provider-correlated units; whole-area interpretation provenance unresolved",
                    "unit34_correction": "Provider confirmed clay; no depth or source-cell change"}

def summarize(rows):
    eligible = [r for r in rows if r["eligible"]]
    coords = {(r["borehole"], r["x"], r["y"]) for r in rows}
    return {"all_segments": len(rows), "eligible_contacts": len(eligible),
            "holes_all": len({r["borehole"] for r in rows}),
            "holes_eligible": len({r["borehole"] for r in eligible}),
            "units_all": len({r["unit"] for r in rows}),
            "units_eligible": len({r["unit"] for r in eligible}),
            "eligible_per_unit": dict(sorted(Counter(r["unit"] for r in eligible).items())),
            "duplicate_coordinate_holes": len(coords) - len({(x, y) for _, x, y in coords}),
            "depth_range_eligible": [min(r["depth"] for r in eligible), max(r["depth"] for r in eligible)]}

def main():
    destination = BASE / "evidence/v2"
    destination.mkdir(parents=True, exist_ok=True)
    rows, audit = [], {}
    for name, loader in (("zhengzhou", load_zhengzhou), ("rvg", load_rvg)):
        data, provenance = loader()
        rows.extend(data)
        audit[name] = {**provenance, **summarize(data)}
    if len({r["target_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate target IDs")
    fields = list(rows[0])
    for name, records in (("all_contacts.csv", rows), ("eligible_contacts.csv", [r for r in rows if r["eligible"]])):
        with (destination / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (destination / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {f: v[f] for f in ("holes_all", "holes_eligible", "eligible_contacts", "units_eligible", "depth_range_eligible")} for k, v in audit.items()}, indent=2))

if __name__ == "__main__":
    main()

