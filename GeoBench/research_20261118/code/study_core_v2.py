"""Fixed, response-isolated conditional interface prediction and scoring."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

METHODS = ("nearest", "idw1", "idw2", "ridge", "forest")
SEED = 20261118

def assign_folds(rows, n, kind):
    holes = {}
    for row in rows:
        xy = (float(row["x"]), float(row["y"]))
        if row["borehole"] in holes and holes[row["borehole"]] != xy:
            raise ValueError("Borehole coordinates are inconsistent")
        holes[row["borehole"]] = xy
    # A shared collar is one grouping unit, including distinct IDs at that collar.
    xy_groups = {}
    for hole in sorted(holes):
        xy_groups.setdefault(holes[hole], []).append(hole)
    coords = list(xy_groups)
    if len(coords) < n:
        raise ValueError("Too few distinct collars for requested folds")
    if kind == "spatial":
        labels = KMeans(n_clusters=n, random_state=SEED, n_init=10).fit_predict(np.array(coords))
    elif kind == "random":
        order = np.random.default_rng(SEED).permutation(len(coords))
        labels = np.empty(len(coords), dtype=int)
        labels[order] = np.arange(len(coords)) % n
    else:
        raise ValueError("Unknown partition kind")
    if len(set(labels)) != n:
        raise ValueError("An empty fold was produced")
    return {hole: int(label) for xy, label in zip(coords, labels) for hole in xy_groups[xy]}

def predict_unit(x, y, query, method):
    distance = np.linalg.norm(query[:, None, :] - x[None, :, :], axis=2)
    nearest = distance.min(axis=1)
    # Baselines all share the exact same donors and finite-output support.
    if method == "nearest":
        pred = y[distance.argmin(axis=1)]
    elif method in ("idw1", "idw2"):
        power = 1 if method == "idw1" else 2
        weights = np.maximum(distance, 1e-12) ** (-power)
        weights /= weights.sum(axis=1, keepdims=True)
        pred = weights @ y
        for i in np.where(nearest <= 1e-10)[0]:
            pred[i] = y[distance[i] <= 1e-10].mean()
    elif method == "ridge":
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=1.).fit(scaler.transform(x), y)
        pred = model.predict(scaler.transform(query))
    elif method == "forest":
        model = RandomForestRegressor(n_estimators=128, min_samples_leaf=3,
                                      max_features=1., random_state=SEED, n_jobs=1).fit(x, y)
        pred = model.predict(query)
    else:
        raise ValueError("Unknown method")
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite supported prediction")
    return np.maximum(pred, 0.), nearest

def predict(train, queries, method):
    donors = defaultdict(list)
    targets = defaultdict(list)
    for row in train:
        donors[row["unit"]].append(row)
    for index, row in enumerate(queries):
        targets[row["unit"]].append(index)
    pred, scale, nearest = [np.full(len(queries), np.nan) for _ in range(3)]
    # Query responses and metadata never enter any estimator or normalization.
    for unit, indices in targets.items():
        if unit not in donors:
            continue
        rows = donors[unit]
        x = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
        y = np.array([r["depth"] for r in rows], dtype=float)
        q = np.array([[queries[i]["x"], queries[i]["y"]] for i in indices], dtype=float)
        values, distances = predict_unit(x, y, q, method)
        pred[indices], nearest[indices] = values, distances
        scale[indices] = max(float(np.std(y)), 1.)
    return pred, scale, nearest

def hole_rmse(rows, pred):
    errors = defaultdict(list)
    for row, value in zip(rows, pred):
        if np.isfinite(value):
            errors[row["borehole"]].append((float(row["depth"]) - value) ** 2)
    if not errors:
        return float("inf")
    return float(np.mean([np.sqrt(np.mean(v)) for v in errors.values()]))

def radius(scores):
    maxima = {}
    for row in scores:
        value = float(row["score"])
        if not np.isfinite(value):
            raise ValueError("Calibration scores must be finite")
        maxima[row["borehole"]] = max(value, maxima.get(row["borehole"], 0.))
    values = sorted(maxima.values())
    rank = math.ceil((len(values) + 1) * .9)
    return values[rank - 1] if rank <= len(values) else float("inf")

def crossfit(rows, folds):
    outputs = {}
    for method in METHODS:
        predictions = np.full(len(rows), np.nan)
        scales = np.full(len(rows), np.nan)
        for fold in sorted(set(folds.values())):
            train = [r for r in rows if folds[r["borehole"]] != fold]
            indices = [i for i, r in enumerate(rows) if folds[r["borehole"]] == fold]
            query = [rows[i] for i in indices]
            values, scale, _ = predict(train, query, method)
            predictions[indices], scales[indices] = values, scale
        scores = [{"borehole": r["borehole"], "score": abs(r["depth"]-p)/s}
                  for r, p, s in zip(rows, predictions, scales) if np.isfinite(p)]
        outputs[method] = {"rmse": hole_rmse(rows, predictions),
                           "radius": radius(scores), "oof_targets": int(np.isfinite(predictions).sum()),
                           "calibration_holes": len({r["borehole"] for r in scores}),
                           "predictions": predictions, "scales": scales}
    masks = [np.isfinite(v["predictions"]) for v in outputs.values()]
    if not all(np.array_equal(masks[0], other) for other in masks[1:]):
        raise ValueError("Inner estimator support differs")
    return outputs

def support_counts(records):
    counts = defaultdict(lambda: [0, 0])
    for row in records:
        counts[row["borehole"]][0] += 1
        counts[row["borehole"]][1] += int(row["supported"])
    complete = {h for h, (total, supported) in counts.items() if total == supported}
    return counts, complete

def summarize_predictions(records):
    groups = defaultdict(list)
    for row in records:
        if row["supported"]:
            groups[row["borehole"]].append(row)
    metrics = []
    for rows in groups.values():
        error = np.array([r["truth"] - r["prediction"] for r in rows])
        width = np.array([r["upper"] - r["lower"] for r in rows])
        coverage = np.array([r["lower"] <= r["truth"] <= r["upper"] for r in rows])
        score = np.array([r["upper"]-r["lower"] + 20*max(r["lower"]-r["truth"],0)
                          + 20*max(r["truth"]-r["upper"],0) for r in rows])
        metrics.append([np.sqrt(np.mean(error**2)), np.mean(np.abs(error)), np.mean(coverage),
                        np.all(coverage), np.mean(width), np.mean(score), np.mean(np.isfinite(width))])
    means = np.mean(np.array(metrics), axis=0) if metrics else np.full(7, np.nan)
    result = dict(zip(("hole_rmse","hole_mae","hole_mean_coverage","supported_hole_all_covered",
                     "hole_mean_width","interval_score","finite_interval_fraction"), map(float,means)),
                targets=len(records), supported=sum(r["supported"] for r in records), holes_supported=len(groups))
    # Report support and finite-interval denominators separately from conditional metrics.
    counts, complete = support_counts(records)
    finite_coverage = []
    complete_coverage = []
    for hole, rows in groups.items():
        finite = [r for r in rows if np.isfinite(r["upper"] - r["lower"])]
        if finite:
            finite_coverage.append(np.mean([r["lower"] <= r["truth"] <= r["upper"] for r in finite]))
        if hole in complete:
            complete_coverage.append(all(r["lower"] <= r["truth"] <= r["upper"] for r in rows))
    result.update(holes_total=len(counts), holes_no_support=sum(s == 0 for t, s in counts.values()),
                  holes_partial_support=sum(0 < s < t for t, s in counts.values()),
                  holes_complete_support=len(complete),
                  complete_hole_all_covered=float(np.mean(complete_coverage)) if complete_coverage else None,
                  finite_interval_coverage=float(np.mean(finite_coverage)) if finite_coverage else None,
                  finite_interval_holes=len(finite_coverage))
    return result

