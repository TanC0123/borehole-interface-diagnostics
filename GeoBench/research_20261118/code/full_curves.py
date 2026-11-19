"""Threshold-free error, calibration and retention curves from saved predictions."""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/"results/curves_20261119"
LEVELS=tuple(sorted({i/100 for i in range(50,100)} | {.975}))

def read(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(values):return sum(values)/len(values)

def hole_ecdf(pairs):
    counts=Counter(h for h,e in pairs)
    mass=defaultdict(float)
    for hole,error in pairs:
        mass[float(error)]+=1/(len(counts)*counts[hole])
    cumulative=0.;points=[]
    for error,weight in sorted(mass.items()):
        cumulative+=weight;points.append((error,min(cumulative,1.)))
    if points:points[-1]=(points[-1][0],1.)
    return points

def residual_quantile(rows,level,target):
    if not 0<level<1:raise ValueError("Nominal level must lie in (0,1)")
    if not rows:return float("inf")
    if target=="simultaneous":
        maxima={}
        for r in rows:maxima[r["borehole"]]=max(maxima.get(r["borehole"],0.),r["score"])
        scores=sorted(maxima.values());rank=math.ceil((len(scores)+1)*level)
        return scores[rank-1] if rank<=len(scores) else float("inf")
    if target!="marginal":raise ValueError("Unknown target")
    counts=Counter(r["borehole"] for r in rows)
    weighted=sorted((r["score"],1/counts[r["borehole"]]) for r in rows)
    cumulative=0.;threshold=level*len(counts)
    for score,weight in weighted:
        cumulative+=weight
        if cumulative+1e-12>=threshold:return float(score)
    return float(weighted[-1][0])

def interval_summary(rows):
    holes=defaultdict(list)
    for r in rows:holes[r["borehole"]].append(r)
    coverage=[];simultaneous=[];width=[];finite=[]
    for group in holes.values():
        inside=[r["lower"]<=r["truth"]<=r["upper"] for r in group]
        coverage.append(mean(inside));simultaneous.append(all(inside))
        width.append(mean([r["upper"]-r["lower"] for r in group]))
        finite.append(mean([math.isfinite(r["upper"]-r["lower"]) for r in group]))
    return {"hole_mean_coverage":mean(coverage),"supported_hole_all_covered":mean(simultaneous),
            "hole_mean_width":mean(width),"finite_interval_fraction":mean(finite),
            "supported_targets":len(rows),"scored_holes":len(holes)}

def error_curves(predictions):
    output=[]
    # Keep every distinct error, including the deepest-tail observations.
    for dataset,method in sorted({(r["dataset"],r["method"]) for r in predictions}):
        rows=[r for r in predictions if r["dataset"]==dataset and r["method"]==method
              and r["calibration"]=="spatial" and r["interval_target"]=="marginal" and r["supported"]=="True"]
        pairs=[(r["borehole"],abs(float(r["truth"])-float(r["prediction"]))) for r in rows]
        for error,fraction in hole_ecdf(pairs):
            output.append({"dataset":dataset,"method":method,"absolute_error_m":error,
                           "hole_balanced_cumulative_probability":fraction,
                           "supported_targets":len(rows),"scored_holes":len({h for h,e in pairs})})
    return output

def calibration_curves(predictions,kriging):
    summary=[];radii=[]
    for dataset in sorted({r["dataset"] for r in predictions}):
        for method in ("idw2","kriging"):
            for kind in ("random","spatial"):
                folded={}
                for fold in range(5):
                    oof=kriging if method=="kriging" else read(BASE/f"results/GB-R3-B/{dataset}_fold{fold}_oof.csv")
                    folded[fold]=[{"borehole":r["borehole"],"score":abs(float(r["truth"])-float(r["prediction"]))/float(r["scale"])}
                        for r in oof if r["dataset"]==dataset and int(r["outer_fold"])==fold
                        and r["method"]==method and r["validation"]==kind and r["supported"]=="True"]
                targets=[r for r in predictions if r["dataset"]==dataset and r["method"]==method
                    and r["calibration"]==kind and r["interval_target"]=="marginal" and r["supported"]=="True"]
                # Re-estimate only residual quantiles; no estimator sees new responses.
                for target in ("marginal","simultaneous"):
                    for level in LEVELS:
                        qs={fold:residual_quantile(scores,level,target) for fold,scores in folded.items()}
                        radii.extend({"dataset":dataset,"method":method,"calibration":kind,"interval_target":target,
                                      "nominal_level":level,"outer_fold":f,"radius":q} for f,q in qs.items())
                        scored=[]
                        for r in targets:
                            value,scale=float(r["prediction"]),float(r["scale"]);q=qs[int(r["outer_fold"])]
                            scored.append({"borehole":r["borehole"],"truth":float(r["truth"]),
                                           "lower":max(0.,value-q*scale),"upper":value+q*scale})
                        summary.append({"dataset":dataset,"method":method,"calibration":kind,
                                        "interval_target":target,"nominal_level":level,**interval_summary(scored)})
    return summary,radii

def retention_curves():
    rows=read(BASE/"results/GB-R3-C/risk_predictions.csv")
    population=Counter(r["dataset"] for r in read(BASE/"evidence/v2/eligible_contacts.csv"))
    outputs=[]
    # Each fold's query batch is ranked independently; risk ties use stable target IDs.
    for ds in sorted(population):
        group=[r for r in rows if r["dataset"]==ds]
        for risk in ("distance","scale","learned"):
            ordered={f:sorted([r for r in group if int(r["outer_fold"])==f],key=lambda r:(float(r[risk]),r["target_id"])) for f in range(5)}
            for percent in range(1,101):
                selected=[r for batch in ordered.values() for r in batch[:math.ceil(len(batch)*percent/100)]]
                ids={r["target_id"] for r in selected};deferred=[r for r in group if r["target_id"] not in ids]
                holes=defaultdict(list);err=[]
                for r in selected:
                    e=abs(float(r["truth"])-float(r["prediction"]));err.append(e);holes[r["borehole"]].append(e*e)
                outputs.append({"dataset":ds,"risk":risk,"nominal_retention":percent/100,
                    "retained_targets":len(selected),"supported_targets":len(group),"eligible_targets":population[ds],
                    "retained_fraction_supported":len(selected)/len(group),"retained_fraction_all":len(selected)/population[ds],
                    "scored_holes":len(holes),"target_mae":mean(err),"target_rmse":math.sqrt(mean([e*e for e in err])),
                    "hole_rmse":mean([math.sqrt(mean(v)) for v in holes.values()]),
                    "deferred_target_mae":mean([abs(float(r["truth"])-float(r["prediction"])) for r in deferred]) if deferred else None})
    return outputs

def verify_existing(calibration,retention):
    d=json.loads((BASE/"results/GB-R3-D/summary.json").read_text())
    for r in calibration:
        if r["nominal_level"]!=.9:continue
        prior=d["/".join(r[k] for k in ("dataset","calibration","method","interval_target"))]
        for key in ("hole_mean_coverage","supported_hole_all_covered","hole_mean_width","finite_interval_fraction"):
            if not math.isclose(r[key],prior[key],rel_tol=1e-10,abs_tol=1e-10):raise ValueError((key,r))
    c=read(BASE/"results/GB-R3-C/decision_metrics_within_fold.csv")
    for old in c:
        current=next(r for r in retention if r["dataset"]==old["dataset"] and r["risk"]==old["risk"] and r["nominal_retention"]==float(old["nominal_retention"]))
        for key in ("retained_targets","target_mae","hole_rmse"):
            if not math.isclose(current[key],float(old[key]),rel_tol=1e-10):raise ValueError(key)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[BASE/"results/GB-R3-D/predictions.csv",BASE/"results/GB-R3-D/kriging_oof.csv",
           BASE/"results/GB-R3-D/summary.json",BASE/"results/GB-R3-C/risk_predictions.csv",
           BASE/"results/GB-R3-C/decision_metrics_within_fold.csv",BASE/"evidence/v2/eligible_contacts.csv",
           *sorted((BASE/"results/GB-R3-B").glob("*_oof.csv"))]
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    predictions=read(paths[0]);calibration,radii=calibration_curves(predictions,read(paths[1]))
    errors=error_curves(predictions);retention=retention_curves()
    verify_existing(calibration,retention)
    for name,records in (("error_ecdf.csv",errors),("calibration_curves.csv",calibration),
                         ("calibration_radii.csv",radii),("retention_curves.csv",retention)):write(OUT/name,records)
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[str(path.relative_to(BASE))]:raise ValueError("Source changed")
    report={"user_decision":"No engineering error tolerance; report curves without pass/fail metres",
            "no_refitting":True,"exploratory_display_after_results":True,"ecdf_full_tail":True,
            "nominal_levels":LEVELS,"retention_grid":"1% to 100% in 1% steps within each outer fold",
            "matches_saved_90_percent_and_original_retention_points":True,"input_hashes":hashes,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows":{"ecdf":len(errors),"calibration":len(calibration),"radii":len(radii),"retention":len(retention)}}
    (OUT/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("input_hashes",)},indent=2))
if __name__=="__main__":main()

