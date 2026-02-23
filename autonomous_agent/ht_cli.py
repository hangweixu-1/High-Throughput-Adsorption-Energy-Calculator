import os
from .util import ensure_dir

def build_download_cmd(repo_root, outdir, chemsys, limit, dl_cfg, api_key=None):
    cmd=["python", os.path.join(repo_root, "download_cif.py")]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += ["--chemsys", chemsys, "--outdir", outdir]
    if int(limit)>0:
        cmd += ["--limit", str(int(limit))]
    # filters
    if dl_cfg.get("is_stable"):
        cmd += ["--is-stable"]
    if dl_cfg.get("energy_above_hull_max") is not None:
        cmd += ["--energy-above-hull-max", str(dl_cfg["energy_above_hull_max"])]
    if dl_cfg.get("exclude_elements"):
        cmd += ["--exclude-elements", str(dl_cfg["exclude_elements"])]
    if dl_cfg.get("band_gap_min") is not None:
        cmd += ["--band-gap-min", str(dl_cfg["band_gap_min"])]
    if dl_cfg.get("band_gap_max") is not None:
        cmd += ["--band-gap-max", str(dl_cfg["band_gap_max"])]
    if dl_cfg.get("num_elements_max") is not None:
        cmd += ["--num-elements-max", str(dl_cfg["num_elements_max"])]
    if dl_cfg.get("conventional"):
        cmd += ["--conventional"]
    # always skip-existing to reduce IO if user keeps same outdir
    cmd += ["--skip-existing"]
    return cmd

def build_ht_cmd(repo_root, cif_folder, engine, out_root, stage_cfg):
    cmd=["python", os.path.join(repo_root, "High_throughput.py"), cif_folder, engine, "-o", out_root]
    if stage_cfg.get("resume", True):
        cmd += ["--resume"]
    if stage_cfg.get("miller"):
        cmd += ["--miller", str(stage_cfg["miller"])]
    if stage_cfg.get("supercell"):
        cmd += ["--supercell", str(stage_cfg["supercell"])]
    if stage_cfg.get("adsorbates"):
        cmd += ["--adsorbates", ",".join(stage_cfg["adsorbates"])]
    if stage_cfg.get("max_candidates") is not None:
        cmd += ["--max-candidates", str(stage_cfg["max_candidates"])]
    if stage_cfg.get("kspacing") is not None:
        cmd += ["--kspacing", str(stage_cfg["kspacing"])]
    if engine == "mace":
        if stage_cfg.get("mace_model"):
            cmd += ["--mace-model", stage_cfg["mace_model"]]
        if stage_cfg.get("mace_device"):
            cmd += ["--mace-device", str(stage_cfg["mace_device"])]
        if stage_cfg.get("mace_dtype"):
            cmd += ["--mace-dtype", str(stage_cfg["mace_dtype"])]
        if stage_cfg.get("fmax_mace") is not None:
            cmd += ["--fmax-mace", str(stage_cfg["fmax_mace"])]
        if stage_cfg.get("max_steps") is not None:
            cmd += ["--max-steps", str(stage_cfg["max_steps"])]
        if stage_cfg.get("pre_rank_steps") is not None:
            cmd += ["--pre-rank-steps", str(stage_cfg["pre_rank_steps"])]
        if stage_cfg.get("pre_rank_fmax") is not None:
            cmd += ["--pre-rank-fmax", str(stage_cfg["pre_rank_fmax"])]
    else:
        # gpaw engine
        if stage_cfg.get("fmax_gpaw") is not None:
            cmd += ["--fmax-gpaw", str(stage_cfg["fmax_gpaw"])]
        if stage_cfg.get("max_steps") is not None:
            cmd += ["--max-steps", str(stage_cfg["max_steps"])]
    return cmd
