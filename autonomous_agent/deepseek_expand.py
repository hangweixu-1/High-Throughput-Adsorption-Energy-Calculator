import os, json, re
from .deepseek_client import DeepSeekClient

_RX=re.compile(r"\[.*\]", re.S)

def _valid_chemsys(s):
    # allow tokens like Cu-Zn-O
    if not isinstance(s,str): return False
    s=s.strip()
    if len(s)<3 or len(s)>32: return False
    toks=s.split("-")
    if len(toks)<2 or len(toks)>5: return False
    for t in toks:
        if not re.fullmatch(r"[A-Z][a-z]?", t):
            return False
    return True

def propose_queries(goal_hint, top_queries, max_new, cfg):
    api_key=(cfg.get("env",{}) or {}).get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return []
    ds=(cfg.get("controller",{}) or {}).get("deepseek", {}) or {}
    model=ds.get("model","deepseek-chat")
    client=DeepSeekClient(api_key, model=model)

    sys=("You propose new Materials Project chemsys strings for catalyst screening. "
         "Return ONLY a JSON list of strings like ['Cu-Zn-O','Cu-Ga-O']. "
         "No commentary.")
    user = {
        "goal": goal_hint,
        "top_queries": top_queries,
        "max_new": max_new
    }
    out=client.chat([
        {"role":"system","content":sys},
        {"role":"user","content":json.dumps(user, ensure_ascii=False)}
    ], temperature=0.3, max_tokens=800)
    m=_RX.search(out)
    if not m:
        return []
    try:
        arr=json.loads(m.group(0))
    except Exception:
        return []
    good=[]
    for s in arr:
        if _valid_chemsys(s):
            good.append(s.strip())
    # dedup keep order
    seen=set(); out2=[]
    for s in good:
        if s not in seen:
            out2.append(s); seen.add(s)
    return out2[:max_new]
