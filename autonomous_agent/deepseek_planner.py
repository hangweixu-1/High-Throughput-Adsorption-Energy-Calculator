import os, json, re
from .deepseek_client import DeepSeekClient

_JSON_RX=re.compile(r"\{.*\}", re.S)

def goal_to_config(goal, base_cfg):
    api_key=(base_cfg.get("env",{}) or {}).get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    model=(base_cfg.get("controller",{}).get("deepseek",{}) or {}).get("model","deepseek-chat")
    client=DeepSeekClient(api_key, model=model)
    sys=("You are a workflow planner. Output ONLY valid JSON with full config. "
         "Modify only: queries.seed_families, ranking.score_terms, stage1.adsorbates, budget fields.")
    msgs=[
        {"role":"system","content":sys},
        {"role":"user","content":"GOAL:\n"+goal},
        {"role":"user","content":"BASE_CONFIG_JSON:\n"+json.dumps(base_cfg, ensure_ascii=False)}
    ]
    out=client.chat(msgs, temperature=0.1, max_tokens=1800)
    m=_JSON_RX.search(out)
    if not m:
        raise RuntimeError("Planner did not return JSON")
    cfg=json.loads(m.group(0))
    return cfg
