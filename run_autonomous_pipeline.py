#!/usr/bin/env python3
import argparse
from autonomous_agent.config import load_config, merge_config_with_env
from autonomous_agent.pipeline import run_pipeline
from autonomous_agent.util import print_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--slurm", action="store_true")
    args = ap.parse_args()

    cfg = merge_config_with_env(load_config(args.config))
    if args.dry_run:
        print("=== DRY RUN ===")
        print_json(cfg)
        return 0
    return run_pipeline(cfg, use_slurm=args.slurm)

if __name__ == "__main__":
    raise SystemExit(main())
