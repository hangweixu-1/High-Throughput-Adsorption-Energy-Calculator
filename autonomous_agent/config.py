import json, os
from copy import deepcopy

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_config_with_env(cfg):
    cfg = deepcopy(cfg)
    cfg.setdefault("env", {})
    if os.getenv("MP_API_KEY"):
        cfg["env"]["MP_API_KEY"] = os.getenv("MP_API_KEY")
    if os.getenv("DEEPSEEK_API_KEY"):
        cfg["env"]["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
    if os.getenv("MACE_MODEL_PATH"):
        if cfg.get("stage1", {}).get("engine") == "mace" and not cfg["stage1"].get("mace_model"):
            cfg["stage1"]["mace_model"] = os.getenv("MACE_MODEL_PATH")
    return cfg
