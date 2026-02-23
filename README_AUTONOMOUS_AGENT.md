# Autonomous Hybrid Agent Wrapper (ADD-ONLY) — v2.0

> ✅ 重要：该补丁 **不改动** 你 GitHub 仓库中的 `download_cif.py` / `High_throughput.py`。  
> 只是在仓库根目录新增 `autonomous_agent/` + 若干入口脚本，通过 subprocess / (可选) SLURM 调用你原脚本，实现：
>
> - **预算耗尽自动停止**（按 `High_throughput.py` 的 `global_checkpoint.json` 里 `completed_files` 计数）
> - **断点续跑**（`workspace/pipeline_state.json` + 原脚本 `--resume`）
> - **工程自愈**：识别 OOM/timeout/SCF/NAN 等错误 → 安全降级参数重试（白名单）
> - **主动学习**：Contextual bandit（LinUCB / UCB / Thompson）
> - **多保真**：Stage-1(MACE) 粗筛 → TopK（可 diverse）→ Stage-2(GPAW) 精炼 → (可选) 线性校准并输出最终排名
> - **最终交付**：`ranked_structures.csv` + `dashboard.html` + Top candidates CIF/XYZ 导出

---

## 1) 解压安装（放到 repo 根目录）
把 zip 解压到你仓库根目录（与 `download_cif.py` 同级）：

```bash
unzip -o HTAEC_autonomous_hybrid_agent_patch_v2_0.zip
```

---

## 2) 必备环境变量
```bash
export MP_API_KEY="你的MP key"

# Stage-1 若用 MACE，需要模型路径（二选一）
export MACE_MODEL_PATH="/path/to/your.model"
# 或者在 config 里 stage1.mace_model 填路径

# 可选：启用 DeepSeek 作为“扩展 chemsys 的规划器”
export DEEPSEEK_API_KEY="你的deepseek key"
```

---

## 3) 运行（本地/单机）
```bash
python run_autonomous_pipeline.py -c examples/autonomous_config.example.json
```

先 dry-run 看计划：
```bash
python run_autonomous_pipeline.py -c examples/autonomous_config.example.json --dry-run
```

---

## 4) 运行（SLURM 集群，可选）
在 config 里把 `slurm.enabled=true`，然后：

```bash
python run_autonomous_pipeline.py -c my_config.json --slurm
```

---

## 5) 输出在哪里看？
默认 workspace：`workspace_budget/`

最终候选输出：
- `workspace_budget/reports/final/ranked_structures.csv`
- `workspace_budget/reports/final/dashboard.html`
- `workspace_budget/reports/final/top_candidates_cifs/`

Stage-1 / Stage-2 原始结果：
- `workspace_budget/ht_stage1/`（对应 `High_throughput.py -o ...` 输出）
- `workspace_budget/ht_stage2/`

---

## 6) 为什么 v2.0 是“关键修复版”？
v2.0 **完全对齐你 repo 的 CLI**：
- `download_cif.py` 用 `--outdir` + `--limit`
- `High_throughput.py` 用 **位置参数**：`python High_throughput.py <cif_folder> <engine> -o <output> ...`
- `--miller` / `--supercell` / `--adsorbates` 都是 **逗号分隔字符串**（如 `1,1,0`）

