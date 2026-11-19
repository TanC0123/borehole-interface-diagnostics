"""Exploratory training-only risk ranking; retain full-population results."""
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BASE=Path(__file__).resolve().parents[1]
B=BASE/"results/GB-R3-B"
OUT=BASE/"results/GB-R3-C"

def support_features(train,query):
    groups=defaultdict(list)
    for row in train:groups[row["unit"]].append(row)
    output=[]
    for row in query:
        donors=groups[row["unit"]]
        nearest=min(math.hypot(row["x"]-r["x"],row["y"]-r["y"]) for r in donors)
        scale=max(float(np.std([r["depth"] for r in donors])),1.)
        output.append([math.log1p(nearest),math.log1p(len(donors)),math.log(scale)])
    return np.array(output)

def retain(risk,fraction,ids):
    if not 0<fraction<=1:raise ValueError("Invalid fraction")
    order=sorted(range(len(risk)),key=lambda i:(risk[i],ids[i]))
    mask=np.zeros(len(risk),dtype=bool)
    mask[order[:math.ceil(len(risk)*fraction)]]=True
    return mask

def read(path):
    with path.open() as handle:return list(csv.DictReader(handle))

def write(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def metrics(rows,mask,total):
    selected=[r for r,k in zip(rows,mask) if k]
    rejected=[r for r,k in zip(rows,mask) if not k]
    errors=[abs(r["truth"]-r["prediction"]) for r in selected]
    by_hole=defaultdict(list)
    for r in selected:by_hole[r["borehole"]].append((r["truth"]-r["prediction"])**2)
    return {"retained_targets":len(selected),"supported_targets":len(rows),"all_eligible_targets":total,
            "retained_fraction_supported":len(selected)/len(rows),"retained_fraction_all":len(selected)/total,
            "holes_retained":len(by_hole),"target_mae":float(np.mean(errors)) if errors else None,
            "target_rmse":float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "hole_rmse":float(np.mean([np.sqrt(np.mean(e)) for e in by_hole.values()])) if by_hole else None,
            "rejected_target_mae":float(np.mean([abs(r["truth"]-r["prediction"]) for r in rejected])) if rejected else None}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=read(BASE/"evidence/v2/eligible_contacts.csv")
    for r in raw:
        for k in ("x","y","depth"):r[k]=float(r[k])
    source={r["target_id"]:r for r in raw}
    prediction_rows=[r for r in read(B/"predictions.csv") if r["method"]=="idw2" and r["calibration"]=="spatial" and r["supported"]=="True"]
    split_rows=read(B/"split_assignments.csv")
    risk_rows=[]
    # Each risk model learns only from residuals within its own outer training set.
    for dataset in sorted({r["dataset"] for r in raw}):
        dataset_rows=[r for r in raw if r["dataset"]==dataset]
        outer={r["borehole"]:int(r["assigned_fold"]) for r in split_rows if r["dataset"]==dataset and r["partition"]=="outer"}
        for fold in range(5):
            training=[r for r in dataset_rows if outer[r["borehole"]]!=fold]
            oof=[r for r in read(B/f"{dataset}_fold{fold}_oof.csv") if r["method"]=="idw2" and r["validation"]=="spatial" and r["supported"]=="True"]
            inner={r["borehole"]:int(r["assigned_fold"]) for r in split_rows if r["dataset"]==dataset and r["partition"]=="spatial" and int(r["outer_fold"])==fold}
            features=np.empty((len(oof),3))
            for inner_fold in range(4):
                indices=[i for i,r in enumerate(oof) if inner[r["borehole"]]==inner_fold]
                donors=[r for r in training if inner[r["borehole"]]!=inner_fold]
                features[indices]=support_features(donors,[source[oof[i]["target_id"]] for i in indices])
            response=np.log1p([abs(float(r["truth"])-float(r["prediction"])) for r in oof])
            counts=Counter(r["borehole"] for r in oof)
            weights=np.array([1/counts[r["borehole"]] for r in oof])
            model=RandomForestRegressor(n_estimators=128,min_samples_leaf=20,random_state=20261118,n_jobs=1)
            model.fit(features,response,sample_weight=weights)
            target=[r for r in prediction_rows if r["dataset"]==dataset and int(r["outer_fold"])==fold]
            query=[source[r["target_id"]] for r in target]
            test_features=support_features(training,query)
            risks={"distance":test_features[:,0],"scale":test_features[:,2],"learned":model.predict(test_features)}
            train_risk={"distance":features[:,0],"scale":features[:,2],"learned":model.predict(features)}
            thresholds={kind:{str(q):float(np.quantile(v,q)) for q in (.5,.75,.9,1.)} for kind,v in train_risk.items()}
            for i,row in enumerate(target):
                risk_rows.append({"dataset":dataset,"outer_fold":fold,"target_id":row["target_id"],
                                  "borehole":row["borehole"],"truth":float(row["truth"]),"prediction":float(row["prediction"]),
                                  **{kind:float(v[i]) for kind,v in risks.items()},
                                  **{f"{kind}_threshold_{q}":value for kind,values in thresholds.items() for q,value in values.items()}})
    reports=[]
    for dataset in sorted({r["dataset"] for r in raw}):
        rows=[r for r in risk_rows if r["dataset"]==dataset]
        total=sum(r["dataset"]==dataset for r in raw)
        for kind in ("distance","scale","learned"):
            for fraction in (.5,.75,.9,1.):
                batch=retain([r[kind] for r in rows],fraction,[r["target_id"] for r in rows])
                threshold=np.array([r[kind]<=r[f"{kind}_threshold_{fraction}"] for r in rows])
                for mode,mask in (("batch_ranking",batch),("training_threshold",threshold)):
                    reports.append({"dataset":dataset,"risk":kind,"nominal_retention":fraction,"decision":mode,
                                    **metrics(rows,mask,total)})
    write(OUT/"risk_predictions.csv",risk_rows);write(OUT/"decision_metrics.csv",reports)
    files=[BASE/"protocol_C.md",Path(__file__),BASE/"evidence/v2/eligible_contacts.csv",
           B/"predictions.csv",B/"split_assignments.csv",*sorted(B.glob("*_oof.csv"))]
    (OUT/"manifest.json").write_text(json.dumps({"hashes":{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       "method":"Exploratory risk ranking; no calibration guarantee","status":"complete"},indent=2)+"\n")
    print(json.dumps([r for r in reports if r["decision"]=="batch_ranking" and r["nominal_retention"] in (.75,1.)],indent=2))

if __name__=="__main__":main()

