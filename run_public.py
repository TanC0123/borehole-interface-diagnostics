"""Portable public-data entry point; outputs go to a new directory."""
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
GB=ROOT/"GeoBench/research_20261118"
P1=ROOT/"核心论文一/research_20261118"
sys.path[:0]=[str(GB/"code"),str(P1/"code")]

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def verify():
    from study_core_v2 import summarize_predictions
    reports={}
    for version in ("GB-R3-B","GB-R3-D","GB-R3-E"):
        run=GB/"results"/version;rows=read(run/"predictions.csv")
        saved=json.loads((run/"summary.json").read_text())
        for r in rows:
            for k in ("truth","prediction","lower","upper"):r[k]=float(r[k])
            r["supported"]=r["supported"]=="True"
            if "selected" in r:r["selected"]=r["selected"]=="True"
        for key,expected in saved.items():
            parts=key.split("/")
            if version=="GB-R3-B":
                ds,kind,method=parts
                selected=[r for r in rows if r["calibration"]==kind and (r["selected"] if method=="selected" else r["method"]==method)]
            elif version=="GB-R3-D":
                ds,kind,method,target=parts
                selected=[r for r in rows if r["calibration"]==kind and r["method"]==method and r["interval_target"]==target]
            else:
                ds,method,variant=parts
                selected=[r for r in rows if r["method"]==method and r["variant"]==variant]
            actual=summarize_predictions(selected)
            for metric in ("hole_rmse","hole_mean_coverage","hole_mean_width"):
                if not math.isclose(actual[metric],expected[metric],rel_tol=1e-10,abs_tol=1e-10):raise ValueError((key,metric))
        reports[version]=len(saved)
    print(json.dumps({"reference_summaries_match":reports,"private_inputs_required":False},indent=2))

def rerun_fold(fold,out):
    from run_interface_study_v2 import run_fold,write_csv
    data=read(GB/"evidence/v2/eligible_contacts.csv")
    for r in data:
        for k in ("x","y","depth"):r[k]=float(r[k])
    outer={r["borehole"]:int(r["assigned_fold"]) for r in read(GB/"results/GB-R3-B/split_assignments.csv") if r["partition"]=="outer"}
    pred,oof,splits,selection=run_fold("rvg",data,outer,fold)
    expected=[r for r in read(GB/"results/GB-R3-B/predictions.csv") if int(r["outer_fold"])==fold]
    indexed={(r["method"],r["calibration"],r["target_id"]):r for r in expected}
    for r in pred:
        old=indexed[(r["method"],r["calibration"],r["target_id"])]
        if r["supported"]!=(old["supported"]=="True"):raise ValueError("support changed")
        if r["supported"] and not math.isclose(r["prediction"],float(old["prediction"]),rel_tol=1e-9,abs_tol=1e-8):raise ValueError("prediction mismatch")
    write_csv(out/"predictions.csv",pred);write_csv(out/"oof.csv",oof)
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print(json.dumps({"fold":fold,"rows":len(pred),"prediction_match":True}))

def constraints(fold,out):
    import constraint_study as module
    data=read(GB/"evidence/v2/eligible_contacts.csv")
    for r in data:
        for k in ("x","y","depth"):r[k]=float(r[k])
    outer={r["borehole"]:int(r["assigned_fold"]) for r in read(GB/"results/GB-R3-B/split_assignments.csv") if r["partition"]=="outer"}
    full,scored,structure,certs,graph=module.run_fold("rvg",data,outer,fold)
    expected={(r["borehole"],r["unit"],r["method"]):float(r["prediction"]) for r in read(P1/"results/P1-R3-G/full_predictions.csv") if int(r["fold"])==fold}
    for r in full:
        if not math.isclose(r["prediction"],expected[(r["borehole"],r["unit"],r["method"])],rel_tol=1e-9,abs_tol=1e-8):raise ValueError("constraint mismatch")
    for name,rows in (("full_predictions.csv",full),("scored.csv",scored),("certificates.csv",certs)):module.write_rows(out/name,rows)
    print(json.dumps({"fold":fold,"constraint_rows":len(full),"prediction_match":True}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("verify","rerun-fold","constraints"))
    parser.add_argument("--fold",type=int,choices=range(5),default=0)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.task=="verify":verify();return
    if args.output is None:parser.error("--output is required for a rerun")
    if args.output.exists():parser.error("Use a new output directory to preserve reference results")
    args.output.mkdir(parents=True)
    if args.task=="rerun-fold":rerun_fold(args.fold,args.output)
    else:constraints(args.fold,args.output)
if __name__=="__main__":main()

