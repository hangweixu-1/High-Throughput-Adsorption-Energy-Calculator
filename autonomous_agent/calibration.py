import json, math
from .results import load_structure_records
from .ranking import rank_structures

def _linfit(x,y):
    n=len(x)
    if n<2: 
        return None
    mx=sum(x)/n; my=sum(y)/n
    sxx=sum((xi-mx)**2 for xi in x)
    if sxx<=1e-12:
        a=1.0; b=my-mx
    else:
        sxy=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
        a=sxy/sxx
        b=my-a*mx
    # r2
    sst=sum((yi-my)**2 for yi in y)
    ssr=sum((yi-(a*xi+b))**2 for xi,yi in zip(x,y))
    r2=1.0-ssr/max(sst,1e-12)
    return a,b,r2

def calibrate_stage1_with_stage2(stage1_out, stage2_out, score_terms, clamp_slope=(0.5,1.5), min_points=6):
    s1=load_structure_records(stage1_out)
    s2=load_structure_records(stage2_out)
    common=set(s1.keys()) & set(s2.keys())
    models={}
    for t in score_terms:
        key=t["key"]
        xs=[]; ys=[]
        for sid in common:
            if key in s1[sid] and key in s2[sid]:
                xs.append(float(s1[sid][key]))
                ys.append(float(s2[sid][key]))
        if len(xs)<min_points:
            continue
        fit=_linfit(xs,ys)
        if not fit:
            continue
        a,b,r2=fit
        lo,hi=clamp_slope
        a=max(lo, min(hi, a))
        models[key]={"a":a,"b":b,"n":len(xs),"r2":r2}
    return models

def apply_calibration(stage1_records, stage2_records, models):
    # stage2 wins if present, else calibrated stage1
    out={}
    for sid,rec in stage1_records.items():
        rr=dict(rec)
        for key,m in models.items():
            if sid in stage2_records and key in stage2_records[sid]:
                rr[key]=float(stage2_records[sid][key])
            elif key in rr:
                rr[key]=float(m["a"])*float(rr[key])+float(m["b"])
        out[sid]=rr
    # include stage2-only records
    for sid,rec in stage2_records.items():
        if sid not in out:
            out[sid]=dict(rec)
    return out
