"""Training-only conditional interface estimates and certified order projection."""
from __future__ import annotations

import json
import math
import platform
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_source import PAPER, ROOT, load_observations, sha256, write_csv, write_json

BASE_METHODS = ("NearestUnit", "IDW", "CoordinateMLP")


def spatial_blocks(metadata, k=5):
    holes = sorted(metadata)
    points = [metadata[b] for b in holes]
    centers = [points[0]]
    for _ in range(1, k):
        centers.append(max(points, key=lambda p: min(sum((p[i] - c[i])**2 for i in (0, 1)) for c in centers)))
    for _ in range(40):
        assignment = [min(range(k), key=lambda j: sum((p[i] - centers[j][i])**2 for i in (0, 1))) for p in points]
        for j in range(k):
            members = [p for p, a in zip(points, assignment) if a == j]
            if members:
                centers[j] = tuple(sum(p[i] for p in members) / len(members) for i in (0, 1))
    return dict(zip(holes, assignment))


def topological_order(nodes, edges):
    if len(nodes) != len(set(nodes)) or not nodes:
        raise ValueError("Nonempty distinct node list required")
    incoming = {n: set() for n in nodes}
    outgoing = {n: set() for n in nodes}
    for a, b in edges:
        if a not in incoming or b not in incoming or a == b:
            raise ValueError("Invalid edge endpoint")
        incoming[b].add(a)
        outgoing[a].add(b)
    order = []
    ready = sorted(n for n in nodes if not incoming[n])
    while ready:
        n = ready.pop(0)
        order.append(n)
        for child in sorted(outgoing[n]):
            incoming[child].remove(n)
            if not incoming[child]:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Cyclic candidate graph; no automatic edge deletion")
    return order


def reduced_edges(nodes, edges):
    order = topological_order(nodes, edges)
    outgoing = {n: set() for n in nodes}
    for a, b in edges:
        outgoing[a].add(b)
    reachable = {n: set() for n in nodes}
    for a in reversed(order):
        for b in outgoing[a]:
            reachable[a].add(b)
            reachable[a].update(reachable[b])
    return [(a, b) for a, b in edges if not any(b in reachable[c] for c in outgoing[a] if c != b)]


def learn_graph(training):
    nodes = sorted({r["unit"] for r in training})
    per_hole = defaultdict(list)
    for r in training:
        per_hole[r["borehole"]].append(r)
    directions = defaultdict(set)
    for bh, rows in per_hole.items():
        rows = sorted(rows, key=lambda r: r["depth"])
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["depth"] < b["depth"]:
                    directions[(a["unit"], b["unit"])].add(bh)
    edges = sorted((a, b) for a, b in directions if (b, a) not in directions)
    topological_order(nodes, edges)
    evidence = [{"above": a, "below": b, "support_n": len(directions[a, b]),
                 "support_holes": sorted(directions[a, b])} for a, b in edges]
    return nodes, edges, evidence


class OrderProjector:
    """Projection onto x>=0 and x_b-x_a>=0, with a checked NNLS dual certificate."""

    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.edges = list(edges)
        topological_order(self.nodes, self.edges)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.reduced = reduced_edges(self.nodes, self.edges)
        matrix = list(np.eye(len(self.nodes)))
        for a, b in self.reduced:
            row = np.zeros(len(self.nodes))
            row[self.index[b]], row[self.index[a]] = 1.0, -1.0
            matrix.append(row)
        self.A = np.array(matrix)

    def project(self, raw):
        if set(raw) != set(self.nodes):
            raise ValueError("Raw prediction must include every graph node exactly once")
        r = np.array([raw[n] for n in self.nodes], dtype=np.float64)
        if not np.all(np.isfinite(r)):
            raise ValueError("Nonfinite raw prediction")
        scale = max(1.0, float(np.max(np.abs(r))))
        z = r / scale
        # Dual: min_{lambda>=0} ||A.T lambda + z||²; primal x=z+A.T lambda.
        # Already feasible inputs have zero dual. Avoid an underdetermined active-set
        # solve for numerical noise at an existing projection's zero-slack faces.
        if float((self.A @ z).min()) >= -1e-12:
            dual = np.zeros(len(self.A))
        else:
            dual, _ = nnls(self.A.T, -z, maxiter=30 * len(self.A))
        x = z + self.A.T @ dual
        slack = self.A @ x
        certificate = {
            "scaled_primal_violation": max(0.0, -float(slack.min())),
            "scaled_dual_violation": max(0.0, -float(dual.min())),
            "scaled_stationarity": float(np.max(np.abs(x - z - self.A.T @ dual))),
            "scaled_complementarity": float(np.max(np.abs(dual * slack))),
        }
        if max(certificate.values()) > 1e-9:
            raise RuntimeError(f"Projection certificate failed: {certificate}")
        result = dict(zip(self.nodes, (x * scale).tolist()))
        violation = max([0.0, *[-v for v in result.values()],
                         *[result[a] - result[b] for a, b in self.edges]])
        if violation > scale * 1e-9:
            raise RuntimeError("Original graph, including reduced edges, is not satisfied")
        certificate.update(max_violation_m=violation, scale_m=scale,
                           max_adjustment_m=float(np.max(np.abs(x * scale - r))))
        return result, certificate


@dataclass
class ConditionalPredictor:
    training: list

    def __post_init__(self):
        self.nodes, self.edges, self.evidence = learn_graph(self.training)
        self.projector = OrderProjector(self.nodes, self.edges)
        self.donors, self.mlps, self.fit_log = {}, {}, []
        for unit in self.nodes:
            rows = sorted((r for r in self.training if r["unit"] == unit), key=lambda r: r["borehole"])
            xy = np.array([[r["x"], r["y"]] for r in rows])
            depth = np.array([r["depth"] for r in rows])
            self.donors[unit] = (xy, depth)
            mu, sd = float(depth.mean()), max(float(depth.std()), 1e-8)
            messages = []
            if len(rows) >= 8:
                model = make_pipeline(StandardScaler(), MLPRegressor(
                    hidden_layer_sizes=(16,), activation="tanh", solver="lbfgs",
                    alpha=0.01, max_iter=500, random_state=0))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(xy, (depth - mu) / sd)
                messages = [str(w.message) for w in caught]
                self.mlps[unit] = (model, mu, sd)
            self.fit_log.append({"unit": unit, "n": len(rows), "mlp_fitted": unit in self.mlps,
                                 "warnings": messages, "donor_holes": [r["borehole"] for r in rows]})

    def predict(self, x, y):
        """Query interface deliberately accepts coordinates only, never target hole metadata."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite query coordinate")
        raw = {m: {} for m in BASE_METHODS}
        for unit in self.nodes:
            xy, depth = self.donors[unit]
            d2 = ((xy - [x, y])**2).sum(axis=1)
            raw["NearestUnit"][unit] = float(depth[np.argmin(d2)])
            zero = d2 < 1e-12
            if zero.any():
                raw["IDW"][unit] = float(depth[zero].mean())
            else:
                w = 1.0 / d2
                raw["IDW"][unit] = float(np.dot(w, depth) / w.sum())
            if unit in self.mlps:
                model, mu, sd = self.mlps[unit]
                raw["CoordinateMLP"][unit] = mu + sd * float(model.predict([[x, y]])[0])
            else:
                raw["CoordinateMLP"][unit] = float(depth.mean())
        predictions, certificates = {}, {}
        for method in BASE_METHODS:
            if not all(math.isfinite(v) for v in raw[method].values()):
                raise RuntimeError("Nonfinite model output; cannot omit failed targets")
            predictions[method] = raw[method]
            predictions[method + "+OrderProjection"], certificates[method] = self.projector.project(raw[method])
        return predictions, certificates


def rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def run_analysis(all_rows, sensitivity=False):
    eligible = [r for r in all_rows if not r["terminal"] and
                (sensitivity or r["geometry_eligible"])]
    metadata = {r["borehole"]: (r["x"], r["y"]) for r in all_rows}
    split = spatial_blocks(metadata)
    predictions, targets, graphs, fits, structural, certificates = [], [], [], [], [], []
    distances = []
    for fold in range(5):
        training = [r for r in eligible if split[r["borehole"]] != fold]
        testing = [r for r in eligible if split[r["borehole"]] == fold]
        training_holes = sorted(b for b in metadata if split[b] != fold)
        testing_holes = sorted(b for b in metadata if split[b] == fold)
        model = ConditionalPredictor(training)
        assert all(set(e["support_holes"]).issubset(training_holes) for e in model.evidence)
        graphs.append({"fold": fold, "training_holes": training_holes, "testing_holes": testing_holes,
                       "nodes": model.nodes, "edges": model.evidence, "reduced_edges": model.projector.reduced})
        fits.extend(dict(r, fold=fold) for r in model.fit_log)
        for r in testing:
            targets.append({"fold": fold, "borehole": r["borehole"], "unit": r["unit"],
                            "source_row": r["source_row"], "truth": r["depth"],
                            "supported": r["unit"] in model.nodes})
        for bh in testing_holes:
            xy = metadata[bh]
            distances.append({"fold": fold, "borehole": bh, "nearest_train_distance_m":
                              min(math.dist(xy, metadata[b]) for b in training_holes)})
            outputs, cert = model.predict(*xy)
            certificates.extend(dict(c, fold=fold, borehole=bh, base_method=m) for m, c in cert.items())
            observed = sorted((r for r in testing if r["borehole"] == bh and r["unit"] in model.nodes), key=lambda r: r["depth"])
            pairs = [(a["unit"], b["unit"]) for i, a in enumerate(observed) for b in observed[i + 1:]]
            for method, output in outputs.items():
                for unit, depth in output.items():
                    predictions.append({"fold": fold, "borehole": bh, "method": method,
                                        "unit": unit, "prediction": depth})
                encoded_bad = sum(output[a] > output[b] + 1e-7 for a, b in model.edges)
                observed_bad = sum(output[a] > output[b] + 1e-7 for a, b in pairs)
                structural.append({"fold": fold, "borehole": bh, "method": method,
                                   "encoded_edges": len(model.edges), "encoded_violations": encoded_bad,
                                   "observed_pairs": len(pairs), "observed_violations": observed_bad,
                                   "negative_depths": sum(d < -1e-7 for d in output.values()),
                                   "emitted_nodes": len(output)})
        print(f"{'sensitivity' if sensitivity else 'primary'} fold {fold}: {len(training)} train / {len(testing)} test contacts", flush=True)
    indexed = {(p["borehole"], p["method"], p["unit"]): p["prediction"] for p in predictions}
    methods = sorted({p["method"] for p in predictions})
    scored, holes, summary = [], [], []
    for method in methods:
        for t in targets:
            key = t["borehole"], method, t["unit"]
            if t["supported"]:
                value = indexed[key]
                if not math.isfinite(value):
                    raise RuntimeError("Missing or nonfinite supported prediction")
                scored.append(dict(t, method=method, prediction=value, error=value - t["truth"]))
        for bh in sorted(metadata):
            bt = [t for t in targets if t["borehole"] == bh]
            sr = [s for s in scored if s["borehole"] == bh and s["method"] == method]
            holes.append({"fold": split[bh], "borehole": bh, "method": method,
                          "n_targets": len(bt), "n_supported": len(sr),
                          "rmse_m": rmse([s["error"] for s in sr]) if sr else None})
        hs = [h for h in holes if h["method"] == method and h["rmse_m"] is not None]
        ss = [s for s in structural if s["method"] == method]
        summary.append({"method": method, "n_targets": len(targets),
                        "n_supported": sum(t["supported"] for t in targets),
                        "coverage": sum(t["supported"] for t in targets) / len(targets),
                        "n_scored_holes": len(hs),
                        "borehole_mean_rmse_m": sum(h["rmse_m"] for h in hs) / len(hs),
                        "encoded_violation_holes": sum(s["encoded_violations"] > 0 for s in ss),
                        "encoded_violations": sum(s["encoded_violations"] for s in ss),
                        "encoded_comparisons": sum(s["encoded_edges"] for s in ss),
                        "observed_violation_holes": sum(s["observed_violations"] > 0 for s in ss),
                        "observed_violations": sum(s["observed_violations"] for s in ss),
                        "observed_comparisons": sum(s["observed_pairs"] for s in ss),
                        "negative_depths": sum(s["negative_depths"] for s in ss),
                        "fold_rmse_m": [sum(h["rmse_m"] for h in hs if h["fold"] == f) /
                                        sum(h["fold"] == f for h in hs) for f in range(5)]})
    support_sets = [{(s["borehole"], s["unit"], s["source_row"]) for s in scored if s["method"] == m} for m in methods]
    assert all(s == support_sets[0] for s in support_sets)
    folder = PAPER / "reproducibility/results" / ("sensitivity" if sensitivity else "primary")
    for name, value in (("predictions", predictions), ("targets", targets), ("scored", scored),
                        ("holes", holes), ("structure", structural), ("certificates", certificates), ("distances", distances)):
        write_csv(folder / (name + ".csv"), value)
    for name, value in (("graphs", graphs), ("fits", fits), ("summary", summary), ("split", split)):
        write_json(folder / (name + ".json"), value)
    return summary


def main():
    rows, _, _ = load_observations()
    result = {}
    for sensitivity in (False, True):
        result["sensitivity" if sensitivity else "primary"] = run_analysis(rows, sensitivity)
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).parent / "protocol.md",
             PAPER / "reproducibility/data/source_audit.json", PAPER / "reproducibility/data/observations.csv"]
    write_json(PAPER / "reproducibility/results/run_manifest.json", {
        "version": "P1-R2.0", "date": "2026-09-18", "analysis_intent": "exploratory_after_historical_results",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "input_code_protocol_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "no_target_metadata_in_prediction_api": True, "hyperparameter_search": False,
        "prediction_files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((PAPER / "reproducibility/results").glob("*/*"))},
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
