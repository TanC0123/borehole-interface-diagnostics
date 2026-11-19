"""Exploratory constraint-evidence study; standard weighted projection, not a new algorithm."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
GB=ROOT/"GeoBench/research_20261118"
sys.path[:0]=[str(ROOT/"核心论文一/reproducibility"),str(GB/"code")]
from reconstruct import learn_graph, reduced_edges, topological_order
from study_core_v2 import predict, hole_rmse

class WeightedProjector:
    def __init__(self,nodes,edges,weights):
        self.nodes=list(nodes);self.edges=list(edges)
        topological_order(self.nodes,self.edges)
        self.index={n:i for i,n in enumerate(self.nodes)}
        if set(weights)!=set(nodes):raise ValueError("Incomplete weights")
        self.weight=np.array([weights[n] for n in nodes],dtype=float)
        if not np.isfinite(self.weight).all() or np.any(self.weight<=0):
            raise ValueError("Weights must be finite and positive")
        self.weight/=self.weight.mean()
        matrix=list(np.eye(len(nodes)))
        for a,b in reduced_edges(nodes,edges):
            row=np.zeros(len(nodes));row[self.index[b]]=1.;row[self.index[a]]=-1.
            matrix.append(row)
        self.A=np.array(matrix)
        self.dual_matrix=self.A.T/np.sqrt(self.weight)[:,None]

    def project(self,raw):
        if set(raw)!=set(self.nodes):raise ValueError("Incomplete input")
        r=np.array([raw[n] for n in self.nodes],dtype=float)
        if not np.isfinite(r).all():raise ValueError("Nonfinite input")
        scale=max(1.,float(np.max(np.abs(r))))
        z=r/scale
        # Diagonal scaling converts the standard weighted dual to nonnegative least squares.
        if np.min(self.A@z)>=-1e-12:
            dual=np.zeros(len(self.A))
        else:
            dual,_=nnls(self.dual_matrix,-np.sqrt(self.weight)*z,maxiter=100*len(self.A))
        x=z+(self.A.T@dual)/self.weight
        slack=self.A@x
        cert={"primal":max(0.,-float(slack.min())),
              "dual":max(0.,-float(dual.min())),
              "stationarity":float(np.max(np.abs(self.weight*(x-z)-self.A.T@dual))),
              "complementarity":float(np.max(np.abs(dual*slack)))}
        if max(cert.values())>1e-8:raise RuntimeError("Invalid certificate: "+str(cert))
        result=dict(zip(self.nodes,(x*scale).tolist()))
        violation=max([0.,*[-v for v in result.values()],*[result[a]-result[b] for a,b in self.edges]])
        if violation>scale*1e-8:raise RuntimeError("Full-graph constraint violation")
        return result,cert

def read_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))

def write_rows(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def training_weights(dataset,fold,nodes):
    records=read_rows(GB/f"results/GB-R3-B/{dataset}_fold{fold}_oof.csv")
    records=[r for r in records if r["validation"]=="spatial" and r["method"]=="idw2" and r["supported"]=="True"]
    original={r["target_id"]:r["unit"] for r in read_rows(GB/"evidence/v2/eligible_contacts.csv")}
    grouped=defaultdict(list)
    for r in records:grouped[original[r["target_id"]]].append((float(r["truth"])-float(r["prediction"]))**2)
    overall=float(np.mean([v for values in grouped.values() for v in values]))
    variance={n:(sum(grouped[n])+5*overall)/(len(grouped[n])+5) for n in nodes}
    return {n:1/max(v,1.) for n,v in variance.items()},variance

def run_fold(dataset,rows,split,fold):
    train=[r for r in rows if split[r["borehole"]]!=fold]
    test=[r for r in rows if split[r["borehole"]]==fold]
    nodes,edges,evidence=learn_graph(train)
    if any(not set(e["support_holes"]) <= {r["borehole"] for r in train} for e in evidence):
        raise ValueError("Graph support leakage")
    weight,variance=training_weights(dataset,fold,nodes)
    trusted=[(e["above"],e["below"]) for e in evidence if e["support_n"]>=3]
    projectors={"hard_all":WeightedProjector(nodes,edges,{n:1. for n in nodes}),
                "weighted_all":WeightedProjector(nodes,edges,weight),
                "weighted_support3":WeightedProjector(nodes,trusted,weight)}
    if dataset=="zhengzhou":
        ordered=sorted(nodes,key=lambda n:int(n[1:]))
        projectors["supplied_order"]=WeightedProjector(nodes,list(zip(ordered,ordered[1:])),{n:1. for n in nodes})
    holes={r["borehole"]:(r["x"],r["y"]) for r in test}
    queries=[{"borehole":h,"unit":n,"x":x,"y":y} for h,(x,y) in sorted(holes.items()) for n in nodes]
    values,_,distances=predict(train,queries,"idw2")
    raw_by_hole=defaultdict(dict)
    for q,v,d in zip(queries,values,distances):raw_by_hole[q["borehole"]][q["unit"]]=(v,d)
    targets=defaultdict(list)
    for r in test:targets[r["borehole"]].append(r)
    counts={n:sum(r["unit"]==n for r in train) for n in nodes}
    full,scored,structure,certificates=[],[],[],[]
    # Project the complete vocabulary before joining any observed test target.
    for hole in sorted(holes):
        raw={n:float(v[0]) for n,v in raw_by_hole[hole].items()}
        outputs={"raw":raw}
        for method,projector in projectors.items():
            outputs[method],cert=projector.project(raw)
            certificates.append({"dataset":dataset,"fold":fold,"borehole":hole,"method":method,**cert})
        observed=sorted([r for r in targets[hole] if r["unit"] in nodes],key=lambda r:r["depth"])
        pairs=[(a["unit"],b["unit"]) for i,a in enumerate(observed) for b in observed[i+1:]]
        for method,output in outputs.items():
            encoded=edges if method=="raw" else projectors[method].edges
            structure.append({"dataset":dataset,"fold":fold,"borehole":hole,"method":method,
                              "observed_pairs":len(pairs),"observed_violations":sum(output[a]>output[b]+1e-7 for a,b in pairs),
                              "encoded_edges":len(encoded),"encoded_violations":sum(output[a]>output[b]+1e-7 for a,b in encoded)})
            for node,value in output.items():
                full.append({"dataset":dataset,"fold":fold,"borehole":hole,"unit":node,"method":method,
                             "prediction":value,"raw":raw[node],"donor_count":counts[node],
                             "training_variance":variance[node]})
            for target in targets[hole]:
                supported=target["unit"] in output
                scored.append({"dataset":dataset,"fold":fold,"borehole":hole,"unit":target["unit"],
                               "target_id":target["target_id"],"method":method,"truth":target["depth"],
                               "prediction":output.get(target["unit"],float("nan")),"supported":supported,
                               "donor_count":counts.get(target["unit"],0)})
    graph={"dataset":dataset,"fold":fold,"nodes":nodes,"evidence":evidence,"variance":variance,
           "training_holes":sorted({r["borehole"] for r in train}),"test_holes":sorted(holes)}
    return full,scored,structure,certificates,graph

def main():
    out=BASE/"results/P1-R3-G"
    out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(GB/"evidence/v2/eligible_contacts.csv")
    for r in rows:
        for k in ("x","y","depth"):r[k]=float(r[k])
    splits=read_rows(GB/"results/GB-R3-B/split_assignments.csv")
    tracked=[BASE/"protocol.md",Path(__file__),GB/"evidence/v2/eligible_contacts.csv",
             GB/"code/study_core_v2.py",ROOT/"核心论文一/reproducibility/reconstruct.py"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    full,scored,structure,certs,graphs,failures=[],[],[],[],[],[]
    for dataset in sorted({r["dataset"] for r in rows}):
        split={r["borehole"]:int(r["assigned_fold"]) for r in splits if r["dataset"]==dataset and r["partition"]=="outer"}
        for fold in range(5):
            try:
                results=run_fold(dataset,[r for r in rows if r["dataset"]==dataset],split,fold)
                for target,values in zip((full,scored,structure,certs),results[:4]):target.extend(values)
                graphs.append(results[4]);print(f"DONE {dataset} {fold}",flush=True)
            except (ValueError,RuntimeError) as error:
                failures.append({"dataset":dataset,"fold":fold,"error":str(error)})
                print(f"FAILED {dataset} {fold}: {error}",flush=True)
    for filename,records in (("full_predictions.csv",full),("scored.csv",scored),("structure.csv",structure),("certificates.csv",certs)):
        if records:write_rows(out/filename,records)
    summaries={}
    for dataset in sorted({r["dataset"] for r in scored}):
        for method in sorted({r["method"] for r in scored if r["dataset"]==dataset}):
            group=[r for r in scored if r["dataset"]==dataset and r["method"]==method]
            r=[{"borehole":v["borehole"],"depth":v["truth"]} for v in group]
            value=hole_rmse(r,np.array([v["prediction"] for v in group]))
            st=[r for r in structure if r["dataset"]==dataset and r["method"]==method]
            summaries[f"{dataset}/{method}"]={"hole_rmse":value,"targets":len(group),
                     "supported":sum(r["supported"] for r in group),"folds":sorted({r["fold"] for r in group}),
                     "observed_violation_holes":sum(r["observed_violations"]>0 for r in st),
                     "observed_violations":sum(r["observed_violations"] for r in st),
                     "encoded_violations":sum(r["encoded_violations"] for r in st)}
    for name,data in (("summary.json",summaries),("graphs.json",graphs),("failures.json",failures),
                      ("manifest.json",{"hashes":hashes,"status":"complete" if not failures else "partial_with_failures","exploratory":True})):
        (out/name).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))

if __name__=="__main__":main()

