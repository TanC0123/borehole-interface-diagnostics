"""Read the private workbook by cell address; retain evidence, never repair cells."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "核心论文一"
LEGACY = ROOT / "论文成品/02_核心论文一_本体约束地层重建"
WORKBOOK = ROOT / "source_file/钻孔及分布资料/航空港分层信息表909.xlsx"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sheet_rows(path):
    """Sparse cells must keep their addresses: a blank B must not move C into B."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
                      for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(NS + "si")]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in relationships}
        sheets = ET.fromstring(archive.read("xl/workbook.xml"))
        output = {}
        for sheet in sheets.iter(NS + "sheet"):
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = targets[rid]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            rows = []
            for row in ET.fromstring(archive.read(member)).iter(NS + "row"):
                values = {"_row": int(row.get("r"))}
                for cell in row.iter(NS + "c"):
                    column = "".join(c for c in cell.get("r") if c.isalpha())
                    value = cell.find(NS + "v")
                    if cell.get("t") == "inlineStr":
                        values[column] = "".join(t.text or "" for t in cell.iter(NS + "t"))
                    elif value is not None:
                        values[column] = shared[int(value.text)] if cell.get("t") == "s" else value.text
                rows.append(values)
            output[sheet.get("name")] = rows
    return output


def load_observations(path=WORKBOOK):
    sheets = sheet_rows(path)
    meta = {}
    for row in sheets["基本情况表"][1:]:
        if not row.get("A"):
            continue
        if row["A"] in meta:
            raise ValueError("Duplicate source borehole metadata")
        meta[row["A"]] = row
    observations, excluded = [], []
    for row in sheets["分层信息表"][1:]:
        if not row.get("A"):
            excluded.append({"source_row": row["_row"], "reason": "no_borehole_id"})
            continue
        if not row.get("C") or not row.get("E"):
            excluded.append({"source_row": row["_row"], "reason": "no_sequence_or_depth"})
            continue
        source = meta[row["A"]]
        sequence = int(row["C"])
        if str(sequence) != row["C"] or sequence < 1:
            raise ValueError("Invalid source sequence code")
        item = {
            "source_row": row["_row"],
            "borehole": "BH-" + hashlib.sha1(row["A"].encode()).hexdigest()[:12],
            "unit": f"S{sequence:03d}",
            "source_sequence": sequence,
            "source_code": row.get("B", ""),
            "x": float(source["D"]), "y": float(source["E"]),
            "surface": float(source["F"]), "hole_depth": float(source["G"]),
            "depth": float(row["E"]), "bottom_elevation": float(row["D"]),
            "row_surface": float(row["F"]), "lithology": row["G"],
        }
        if not all(math.isfinite(item[k]) for k in
                   ("x", "y", "surface", "hole_depth", "depth", "bottom_elevation", "row_surface")):
            raise ValueError("Nonfinite source numeric value")
        observations.append(item)
    by_hole = defaultdict(list)
    for row in observations:
        by_hole[row["borehole"]].append(row)
    terminal = Counter()
    for rows in by_hole.values():
        rows.sort(key=lambda r: r["depth"])
        if len({r["unit"] for r in rows}) != len(rows):
            raise ValueError("Repeated source unit in a borehole requires adjudication")
        delta = rows[-1]["depth"] - rows[-1]["hole_depth"]
        terminal["at_hole_depth" if abs(delta) < 1e-6 else "beyond_hole_depth" if delta > 0 else "before_hole_depth"] += 1
        previous = 0.0
        for index, row in enumerate(rows):
            flags = []
            row["record_start"] = previous
            if row["depth"] <= previous:
                flags.append("nonpositive_record_interval")
            if row["depth"] > row["hole_depth"] + 1e-8:
                flags.append("beyond_reported_hole_depth")
            if abs(row["depth"] + row["bottom_elevation"] - row["row_surface"]) > 0.011:
                flags.append("depth_elevation_identity_conflict")
            if abs(row["surface"] - row["row_surface"]) > 0.011:
                flags.append("surface_metadata_conflict")
            row["terminal"] = index == len(rows) - 1
            row["qc_flags"] = "|".join(flags)
            row["geometry_eligible"] = not row["terminal"] and not flags
            # Presence of a logged interval does not establish that its base was reached.
            row["observation_state"] = "recorded_terminal" if row["terminal"] else "recorded_internal_contact"
            row["absence_target"] = "unavailable"
            previous = row["depth"]
    return observations, excluded, dict(terminal)


def main():
    rows, excluded, terminal = load_observations()
    candidate = LEGACY / "candidate_data_v03"
    manifest = json.loads((candidate / "manifest.json").read_text())
    structure = json.loads((PAPER / "private_report/manuscript_structure.json").read_text())
    hash_checks = {name: sha256(candidate / name) == expected
                   for name, expected in manifest["sha256"].items()}
    manuscript_matches = sha256(LEGACY / "核心论文一_candidate_paper.html") == structure["sha256"]
    if not all(hash_checks.values()) or not manuscript_matches:
        raise ValueError("Phase-1 source version changed; inspect before continuing")
    with (candidate / "full_processed_909.csv").open(encoding="utf-8-sig") as stream:
        legacy = list(csv.DictReader(stream))
    indexed = {(r["borehole"], r["unit"]): r for r in rows}
    legacy_matches = all(
        indexed[(r["blind_borehole_id"], f"S{int(r['seq_code']):03d}")]["depth"] == float(r["depth_bottom"])
        for r in legacy)
    assert legacy_matches
    metadata = {r["borehole"]: (r["x"], r["y"]) for r in rows}
    code_mapping = defaultdict(set)
    for r in rows:
        if r["source_code"]:
            code_mapping[r["source_sequence"]].add(r["source_code"])
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(LEGACY.rglob("*"))
              if path.is_file() and "__pycache__" not in path.parts}
    hashes[str(WORKBOOK.relative_to(ROOT))] = sha256(WORKBOOK)
    report = {
        "date": "2026-09-18", "old_manuscript_unchanged": manuscript_matches,
        "candidate_hash_checks": hash_checks, "legacy_depths_match_raw": legacy_matches,
        "workbook_sha256": sha256(WORKBOOK),
        "raw_records": len(rows), "holes": len(metadata),
        "units": len({r["unit"] for r in rows}), "lithologies": len({r["lithology"] for r in rows}),
        "restored_records": [{k: r[k] for k in ("source_row", "borehole", "unit", "depth", "terminal", "geometry_eligible")}
                             for r in rows if not r["source_code"]],
        "source_sequence_to_code_one_to_one": all(len(v) == 1 for v in code_mapping.values()),
        "source_sequence_is_not_used_as_numerical_model_feature_or_order": True,
        "terminal_relation": terminal, "all_terminal_records": sum(r["terminal"] for r in rows),
        "qc_flag_counts": dict(Counter(flag for r in rows for flag in r["qc_flags"].split("|") if flag)),
        "qc_internal_records": sum(not r["terminal"] and bool(r["qc_flags"]) for r in rows),
        "eligible_contacts": sum(r["geometry_eligible"] for r in rows),
        "quarantined_records": [{k: r[k] for k in ("source_row", "borehole", "unit", "depth", "terminal", "qc_flags")}
                                for r in rows if r["qc_flags"]],
        "duplicate_coordinate_holes": len(metadata) - len(set(metadata.values())),
        "sidetrack_relationship": "No parent/sidetrack field in source sheets; unknown, not verified absent",
        "excluded_source_rows": excluded,
        "source_fields": {name: {k: v for k, v in records[0].items() if k != "_row"}
                          for name, records in sheet_rows(WORKBOOK).items()},
        "legacy_artifact_hashes": hashes,
    }
    write_csv(PAPER / "reproducibility/data/observations.csv", rows)
    write_json(PAPER / "reproducibility/data/source_audit.json", report)
    print(json.dumps({k: report[k] for k in ("raw_records", "holes", "units", "eligible_contacts", "terminal_relation", "qc_flag_counts", "qc_internal_records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
