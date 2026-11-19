"""Geostatistical control and distinct marginal versus simultaneous interval targets."""
import csv
import hashlib
import json
import math
import warnings
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pykrige
from pykrige.ok import OrdinaryKriging
from study_core_v2 import radius,summarize_predictions,hole_rmse

BASE=Path(__file__).resolve().parents[1]
B=BASE/"results/GB-R3-B"
OUT=BASE/"results/GB-R3-D"

def kriging(x,y,q):
    # Duplicate collars must not create a singular system or receive extra weight.
    grouped=defaultdict(list)
    for coord,value in zip(x,y):grouped[tuple(coord)].append(value)
    x=np.array(list(grouped),dtype=float)
    y=np.array([np.mean(v) for v in grouped.values()])
    if len(y)<5 or np.std(y)<1e-8:return np.full(len(q),float(y.mean()))
    center=x.mean(axis=0)
    tx=(x-center)/1000.;tq=(q-center)/1000.
    model=OrdinaryKriging(tx[:,0],tx[:,1],y,variogram_model="linear",nlags=6,
                         weight=False,exact_values=True,pseudo_inv=True,
                         verbose=False,enable_plotting=False)
    result,_=model.execute("points",tq[:,0],tq[:,1],backend="loop",n_closest_points=min(64,len(y)))
    result=np.asarray(result,dtype=float)
    if not np.isfinite(result).all():raise ValueError("Nonfinite kriging output")
    return np.maximum(result,0.)

def marginal_radius(rows):
    count=Counter(r["borehole"] for r in rows)
    if not count:return float("inf")
    # Each borehole contributes total mass one, irrespective of observed target count.
    ordered=sorted([(r["score"],1/count[r["borehole"]]) for r in rows])
    cumulative=0.;threshold=.9*len(count)
    for score,weight in ordered:
        cumulative+=weight
        if cumulative+1e-12>=threshold:return float(score)
    return float(ordered[-1][0])

def predict(train,queries):
    grouped=defaultdict(list);indices=defaultdict(list)
    for r in train:grouped[r["unit"]].append(r)
    for i,r in enumerate(queries):indices[r["unit"]].append(i)
    values,scales=[np.full(len(queries),np.nan) for _ in range(2)]
    for unit,ids in indices.items():
        if unit not in grouped:continue
        donors=grouped[unit]
        x=np.array([[r["x"],r["y"]] for r in donors]);y=np.array([r["depth"] for r in donors])
        q=np.array([[queries[i]["x"],queries[i]["y"]] for i in ids])
        values[ids]=kriging(x,y,q);scales[ids]=max(float(np.std(y)),1.)
    return values,scales

def read(path):
    with path.open() as handle:return list(csv.DictReader(handle))
def write(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read(BASE/"evidence/v2/eligible_contacts.csv")
    for r in rows:
        for k in ("x","y","depth"):r[k]=float(r[k])
    splits=read(B/"split_assignments.csv")
    existing=read(B/"predictions.csv")
    predictions,oof_all,calibration_log,warning_log=[],[],[],[]
    for dataset in sorted({r["dataset"] for r in rows}):
        data=[r for r in rows if r["dataset"]==dataset]
        outer={r["borehole"]:int(r["assigned_fold"]) for r in splits if r["dataset"]==dataset and r["partition"]=="outer"}
        for fold in range(5):
            train=[r for r in data if outer[r["borehole"]]!=fold]
            test=[r for r in data if outer[r["borehole"]]==fold]
            oof=read(B/f"{dataset}_fold{fold}_oof.csv")
            # Only the new baseline is fitted; existing point estimates stay frozen.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outer_pred,outer_scale=predict(train,test)
                for kind in ("random","spatial"):
                    inner={r["borehole"]:int(r["assigned_fold"]) for r in splits if r["dataset"]==dataset and r["partition"]==kind and int(r["outer_fold"])==fold}
                    for inner_fold in range(4):
                        donors=[r for r in train if inner[r["borehole"]]!=inner_fold]
                        query=[r for r in train if inner[r["borehole"]]==inner_fold]
                        values,scales=predict(donors,query)
                        for r,v,s in zip(query,values,scales):
                            record={"dataset":dataset,"outer_fold":fold,"validation":kind,"method":"kriging",
                                    "target_id":r["target_id"],"borehole":r["borehole"],"inner_fold":inner_fold,
                                    "truth":r["depth"],"prediction":v,"scale":s,"supported":bool(np.isfinite(v))}
                            oof.append(record);oof_all.append(record)
            warning_log.extend({"dataset":dataset,"fold":fold,"message":str(w.message)} for w in caught)
            for method in ("nearest","idw1","idw2","ridge","forest","kriging"):
                for kind in ("random","spatial"):
                    calibration=[r for r in oof if r["method"]==method and r["validation"]==kind and str(r["supported"])=="True"]
                    scores=[{"borehole":r["borehole"],"score":abs(float(r["truth"])-float(r["prediction"]))/float(r["scale"])} for r in calibration]
                    radii={"simultaneous":radius(scores),"marginal":marginal_radius(scores)}
                    for target_type,q in radii.items():
                        calibration_log.append({"dataset":dataset,"fold":fold,"method":method,"calibration":kind,
                                                "interval_target":target_type,"radius":q,"calibration_targets":len(scores),
                                                "calibration_holes":len({r["borehole"] for r in scores})})
                        if method=="kriging":
                            base=[{"target_id":r["target_id"],"borehole":r["borehole"],"unit":r["unit"],"truth":r["depth"],
                                   "prediction":v,"scale":s,"supported":bool(np.isfinite(v))} for r,v,s in zip(test,outer_pred,outer_scale)]
                        else:
                            base=[r for r in existing if r["dataset"]==dataset and int(r["outer_fold"])==fold and r["method"]==method and r["calibration"]==kind]
                        for r in base:
                            supported=str(r["supported"])=="True"
                            value,scale=float(r["prediction"]),float(r["scale"])
                            predictions.append({"dataset":dataset,"outer_fold":fold,"method":method,"calibration":kind,
                                                "interval_target":target_type,"target_id":r["target_id"],"borehole":r["borehole"],
                                                "unit":r["unit"],"truth":float(r["truth"]),"prediction":value,"scale":scale,
                                                "lower":max(0.,value-q*scale) if supported else float("nan"),
                                                "upper":value+q*scale if supported else float("nan"),"supported":supported})
            print(f"DONE {dataset} {fold}",flush=True)
    write(OUT/"predictions.csv",predictions);write(OUT/"kriging_oof.csv",oof_all);write(OUT/"calibration.csv",calibration_log)
    summaries={}
    for dataset,method,kind,target in sorted({(r["dataset"],r["method"],r["calibration"],r["interval_target"]) for r in predictions}):
        group=[r for r in predictions if (r["dataset"],r["method"],r["calibration"],r["interval_target"])==(dataset,method,kind,target)]
        summaries["/".join((dataset,kind,method,target))]=summarize_predictions(group)
    (OUT/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (OUT/"warnings.json").write_text(json.dumps(warning_log,indent=2)+"\n")
    files=[BASE/"protocol_D.md",Path(__file__),BASE/"code/study_core_v2.py",
           BASE/"evidence/v2/eligible_contacts.csv",B/"predictions.csv",B/"split_assignments.csv",*sorted(B.glob("*_oof.csv"))]
    (OUT/"manifest.json").write_text(json.dumps({"hashes":{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "pykrige":pykrige.__version__,"status":"complete","warning_count":len(warning_log)},indent=2)+"\n")
    print(json.dumps({k:v for k,v in summaries.items() if "/kriging/marginal" in k or "/idw2/marginal" in k},indent=2))

if __name__=="__main__":main()

