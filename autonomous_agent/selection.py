import math, random

def select_topk(ranked, k):
    return ranked[:k]

def select_diverse(ranked, k, keys):
    # greedy k-center on specified descriptor keys
    cand=[r for r in ranked if all((key in r and r[key] is not None) for key in keys)]
    if not cand:
        return ranked[:k]
    # normalize
    feats=[]
    for r in cand:
        vec=[float(r[key]) for key in keys]
        feats.append(vec)
    # compute mean/std
    m=[sum(v[i] for v in feats)/len(feats) for i in range(len(keys))]
    s=[]
    for i in range(len(keys)):
        var=sum((v[i]-m[i])**2 for v in feats)/max(len(feats)-1,1)
        s.append(math.sqrt(var) if var>1e-12 else 1.0)
    norm=[[ (v[i]-m[i])/s[i] for i in range(len(keys)) ] for v in feats]
    # start with best (ranked[0]) if present else random
    chosen=[]
    used=set()
    # pick first = lowest rank (cand already in rank order)
    chosen_idx=0
    chosen.append(cand[chosen_idx]); used.add(chosen_idx)
    # distance function
    def dist(a,b):
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(keys))))
    while len(chosen)<min(k,len(cand)):
        best_i=None; best_d=-1
        for i in range(len(cand)):
            if i in used: continue
            d=min(dist(norm[i], norm[j]) for j in used)
            if d>best_d:
                best_d=d; best_i=i
        used.add(best_i)
        chosen.append(cand[best_i])
    return chosen
