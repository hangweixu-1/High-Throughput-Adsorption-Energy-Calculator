#!/usr/bin/env python3
"""Optional: DeepSeek goal -> config -> run (ADD-ONLY)."""
import argparse, json
from autonomous_agent.config import load_config, merge_config_with_env
from autonomous_agent.deepseek_planner import goal_to_config
from autonomous_agent.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--slurm", action="store_true")
    args = ap.parse_args()

    base = merge_config_with_env(load_config(args.template))
    cfg = goal_to_config(args.goal, base)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out_config}")

    if args.run:
        return run_pipeline(cfg, use_slurm=args.slurm)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
