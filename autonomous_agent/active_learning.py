import math, random

def featurize_query(q, dim=128):
    # hashed bag of elements (q like 'Cu-Zn-O')
    elems=[e.strip() for e in q.split("-") if e.strip()]
    x=[0.0]*dim
    for e in elems:
        h=hash(e)%dim
        x[h]+=1.0
    # bias
    x[0]+=1.0
    return x

class LinUCBGlobalDiag:
    """Contextual bandit with diagonal A (ridge) for efficiency."""
    def __init__(self, alpha=1.5, dim=128, l2=1.0):
        self.alpha=float(alpha); self.dim=int(dim)
        self.A=[float(l2)]*self.dim
        self.b=[0.0]*self.dim
    def update(self, arm, reward):
        x=featurize_query(arm, self.dim)
        for i,xi in enumerate(x):
            self.A[i]+=xi*xi
            self.b[i]+=reward*xi
    def score(self, arm):
        x=featurize_query(arm, self.dim)
        mu=0.0; var=0.0
        for i,xi in enumerate(x):
            if xi==0: continue
            theta=self.b[i]/self.A[i]
            mu += theta*xi
            var += (xi*xi)/self.A[i]
        return mu + self.alpha*math.sqrt(max(var,1e-9))

class UCB1:
    def __init__(self, alpha=1.5):
        self.alpha=float(alpha); self.t=0
        self.n={}; self.mean={}
    def update(self, arm, reward):
        self.t+=1
        n=self.n.get(arm,0)+1
        m=self.mean.get(arm,0.0)
        self.mean[arm]=m+(reward-m)/n
        self.n[arm]=n
    def score(self, arm):
        n=self.n.get(arm,0)
        if n==0: return float("inf")
        return self.mean.get(arm,0.0)+self.alpha*math.sqrt(math.log(max(self.t,2))/n)

class ThompsonGaussian:
    def __init__(self):
        self.n={}; self.mean={}; self.m2={}
    def update(self, arm, reward):
        n=self.n.get(arm,0)+1
        delta=reward-self.mean.get(arm,0.0)
        mean=self.mean.get(arm,0.0)+delta/n
        m2=self.m2.get(arm,0.0)+delta*(reward-mean)
        self.n[arm]=n; self.mean[arm]=mean; self.m2[arm]=m2
    def score(self, arm):
        n=self.n.get(arm,0)
        if n<2: return float("inf")
        var=max(self.m2.get(arm,1.0)/(n-1), 1e-6)
        std=math.sqrt(var)/math.sqrt(n)
        return random.gauss(self.mean.get(arm,0.0), std)

def make_agent(cfg):
    al=cfg.get("active_learning", {}) or {}
    alg=(al.get("algorithm","linucb") or "linucb").lower()
    alpha=float(al.get("alpha",1.5))
    if alg=="ucb":
        return UCB1(alpha=alpha)
    if alg=="thompson":
        return ThompsonGaussian()
    return LinUCBGlobalDiag(alpha=alpha)
