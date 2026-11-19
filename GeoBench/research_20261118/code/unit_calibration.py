"""Unit-specific empirical calibration, using frozen training residuals only."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from study_core_v2 import summarize_predictions
BASE=Path(__file__).resolve().parents[1]
B=BASE/"results/GB-R3-B";D=BASE/"results/GB-R3-D";OUT=BASE/"results/GB-R3-E"
def read(p):
    with p.open() as stream:return list(csv.DictReader(stream))
def write(p,rows):
    with p.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source={r["target_id"]:r["unit"] for r in read(BASE/"evidence/v2/eligible_contacts.csv")}
    raw=read(D/"predictions.csv");krig=read(D/"kriging_oof.csv")
    global_q={(r["dataset"],int(r["fold"]),r["method"]):float(r["radius"]) for r in read(D/"calibration.csv")
              if r["calibration"]=="spatial" and r["interval_target"]=="marginal"}
    outputs=[];calibration=[]
    # Class thresholds are estimated from each outer training set's OOF residuals.
    for dataset in ("rvg","zhengzhou"):
        for fold in range(5):
            previous=read(B/f"{dataset}_fold{fold}_oof.csv")
            for method in ("idw2","kriging"):
                oof=krig if method=="kriging" else previous
                oof=[r for r in oof if r["dataset"]==dataset and int(r["outer_fold"])==fold
                     and r["method"]==method and r["validation"]=="spatial" and r["supported"]=="True"]
                groups=defaultdict(list)
                for r in oof:groups[source[r["target_id"]]].append(abs(float(r["truth"])-float(r["prediction"]))/float(r["scale"]))
                radii={}
                for unit,values in groups.items():
                    rank=math.ceil((len(values)+1)*.9)
                    calibrated=len(values)>=20 and rank<=len(values)
                    radii[unit]=(sorted(values)[rank-1] if calibrated else global_q[(dataset,fold,method)],calibrated,len(values))
                    calibration.append({"dataset":dataset,"fold":fold,"method":method,"unit":unit,
                                        "calibration_targets":len(values),"unit_calibrated":calibrated,
                                        "radius":radii[unit][0]})
                targets=[r for r in raw if r["dataset"]==dataset and int(r["outer_fold"])==fold and r["method"]==method
                         and r["calibration"]=="spatial" and r["interval_target"]=="marginal"]
                for r in targets:
                    supported=r["supported"]=="True"
                    q,calibrated,n=radii.get(r["unit"],(global_q[(dataset,fold,method)],False,0))
                    for variant in ("global_marginal","unit_min20"):
                        value=float(r["prediction"]);scale=float(r["scale"])
                        actual=global_q[(dataset,fold,method)] if variant=="global_marginal" else q
                        outputs.append({"dataset":dataset,"outer_fold":fold,"method":method,"variant":variant,
                                        "target_id":r["target_id"],"borehole":r["borehole"],"unit":r["unit"],
                                        "truth":float(r["truth"]),"prediction":value,
                                        "lower":max(0.,value-actual*scale) if supported else float("nan"),
                                        "upper":value+actual*scale if supported else float("nan"),
                                        "supported":supported,"unit_calibrated":calibrated if variant=="unit_min20" else False,
                                        "calibration_targets_unit":n})
    summaries={};strata=[]
    for dataset in ("rvg","zhengzhou"):
        for method in ("idw2","kriging"):
            for variant in ("global_marginal","unit_min20"):
                group=[r for r in outputs if (r["dataset"],r["method"],r["variant"])==(dataset,method,variant)]
                summary=summarize_predictions(group)
                summary["unit_calibrated_targets"]=sum(r["supported"] and r["unit_calibrated"] for r in group)
                summary["pooled_fallback_targets"]=sum(r["supported"] and not r["unit_calibrated"] for r in group)
                summaries["/".join((dataset,method,variant))]=summary
                for unit in sorted({r["unit"] for r in group}):
                    rows=[r for r in group if r["unit"]==unit and r["supported"]]
                    if not rows:continue
                    strata.append({"dataset":dataset,"method":method,"variant":variant,"unit":unit,
                                   "targets":len(rows),"covered":sum(r["lower"]<=r["truth"]<=r["upper"] for r in rows),
                                   "coverage":np.mean([r["lower"]<=r["truth"]<=r["upper"] for r in rows]),
                                   "mean_width":np.mean([r["upper"]-r["lower"] for r in rows]),
                                   "unit_calibrated":sum(r["unit_calibrated"] for r in rows)})
    write(OUT/"predictions.csv",outputs);write(OUT/"unit_metrics.csv",strata);write(OUT/"calibration.csv",calibration)
    (OUT/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    files=[BASE/"protocol_E.md",Path(__file__),BASE/"code/study_core_v2.py",BASE/"evidence/v2/eligible_contacts.csv",
           D/"predictions.csv",D/"calibration.csv",D/"kriging_oof.csv",*sorted(B.glob("*_oof.csv"))]
    (OUT/"manifest.json").write_text(json.dumps({"hashes":{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"status":"complete","exploratory_after_D":True},indent=2)+"\n")
    print(json.dumps({k:{m:v[m] for m in ("hole_mean_coverage","hole_mean_width","interval_score","unit_calibrated_targets","pooled_fallback_targets")} for k,v in summaries.items()},indent=2))
if __name__=="__main__":main()

