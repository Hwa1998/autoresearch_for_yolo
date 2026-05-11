# AutoResearch for YOLO — English README

> This repository is an adaptation of Andrej Karpathy's [karpathy/autoresearch](https://github.com/karpathy/autoresearch). It demonstrates engineering practices for autonomous LLM-driven experiments and applies them to YOLO object detection.

> Note: This project was refactored with assistance from AI tools; you may notice prominent AI-related artifacts in the codebase.

## 0. Project Background

In March 2026 Andrej Karpathy released [autoresearch](https://github.com/karpathy/autoresearch). The core idea is: let an LLM-based agent run overnight experiments in a small but realistic training environment—edit code, train, evaluate, keep or discard changes, and iterate. This repository ports that idea to the YOLO object-detection domain and includes implementations and design choices for LLM patch generation, hyperparameter search, and model-structure exploration.

## 1. Key Features

- Automated Phase1–Phase4 experiment loop: baseline selection → HP search → long-run validation → structure/loss exploration
- Single-model training mode: run a standard long-run training for a specified model YAML (no searching)
- Suggestion-JSON driven runs: pass an experiment suggestion JSON to run a single trial and record results
- LLM-driven code modification: supports DeepSeek / OpenAI / Ali Tongyi / Moonshot / SiliconFlow / Anthropic Claude for automated code patches and verification
- Conservative patching and safe application: git-branch isolation, smoke tests → short run → keep/discard
- Model YAML variant generation: text-level substitutions (e.g., C2f → C2PSA) to explore structural variants
- Focused-TAL: inject metrics via environment variables to adjust TaskAlignedAssigner alignment for certain classes
- Long-running autonomous agent: unattended loop that periodically proposes, applies, and validates patches with logs

## 2. Repository Layout & Core Modules

Project root highlights:

- `train.py` — simplified GPT pretraining script (nanochat-style); editable by the Agent for LLM pretraining experiments
- `prepare.py` — data preparation and evaluation interface (`evaluate_placeholder`); treated as non-editable
- `train_yolov13.py` — YOLOv13 time-budget training wrapper
- `program.md` — Agent taskbook (experiment constraints, evaluation rules, keep/discard logic)
- `configs/autoresearch_config.yaml` — experiment configuration (candidate models, phase parameters, LLM keys, etc.)

`lib/` — shared libraries:

- `lib/config.py` (~20 LOC): YAML config loader with CLI override precedence
- `lib/llm.py` (~80 LOC): unified LLM client adapter (DeepSeek/OpenAI/Anthropic, etc.)
- `lib/training.py` (~60 LOC): training launch, checkpoint discovery, evaluation wrapper
- `lib/models.py` (~130 LOC): model YAML candidate discovery, scale expansion, ranking
- `lib/patches.py` (~200 LOC): diff generation, safety checks, git-branch safe application

`scripts/` — primary scripts:

- `scripts/autoresearch_loop.py` — orchestrator: Phase1–Phase4, single-model training, suggestion mode
- `scripts/run_yolo_train.py` — YOLO training wrapper (hp → CLI → env → tal)
- `scripts/autonomous_agent.py` — long-running daemon for periodic LLM patch proposals and application
- `scripts/llm_patch_agent.py` — generates LLM-driven patches (lib/llm → validate → smoke → short run)
- `scripts/llm_patch_validator.py` — AST + py_compile validation
- `scripts/agent_patch_generator.py` — conservative patch generator (default HP tweaks only)
- `scripts/patch_runner.py` — safe patch applier (git branch → smoke → short run → keep/discard)
- `scripts/model_variant_generator.py` — generate model YAML variants via text substitution

`yolov13/` — YOLOv13 implementation and configs:

- `yolov13/ultralytics/utils/tal.py` — TaskAlignedAssigner + Focused-TAL (env-var injection)
- `yolov13/ultralytics/cfg/models/v13/` — model YAML files

### Phase1–Phase4 Overview

Phase summary:

- Phase1: baseline model screening (short epochs × many candidate models). Example candidates: `yolov13n`, `yolov13s`, `yolov13n_nmsfree`. Rank by mAP@0.5 and keep top-K (`--keep-k`, default 2).
- Phase2: HP random search (short epochs × K models × N rounds). Randomize lr, batch, mosaic, degrees. Produce per-model HP champion.
- Phase3: long-run confirmation (long epochs × K champions). Produce final baseline mAP@0.5 and mAP@0.5-0.95.
- Phase4: LLM-guided exploration: feed Phase1–3 context to LLM; LLM proposes HP and structural experiments; system executes short-run validations; iterate 3–5 rounds and then long-run confirm the best.

Keep/Discard rule: primarily based on short-run mAP@0.5; improvements over historical best are committed, otherwise reverted.

Dataflow sketch:

`autoresearch_config.yaml` + `program.md` → `scripts/autoresearch_loop.py` (orchestrator) → train subprocesses (`run_yolo_train.py`) → `prepare.py` evaluation → append records to `runs/autoresearch/results.jsonl`. Core helpers in `lib/training`, `lib/llm`, `lib/patches`, `lib/models`.

## 3. How to Run

> **Environment**: This project uses [uv](https://docs.astral.sh/uv/) to manage Python virtual environments.
> After cloning, run `uv venv` at the repo root to create `.venv`, then `uv pip install -r requirements.txt` for dependencies.
>
> All examples assume you are in the repo root using Windows PowerShell with the virtual environment activated.
> On Windows, you must set `PYTHONPATH` before running to ensure module imports work correctly:

3.1 Single-model long-run training (terminal):

```powershell
# Use default HPs to long-run train a specified model YAML (e.g. 200 epochs)
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --mode train \
  --model yolov13/ultralytics/cfg/models/v13/yolov13.yaml \
  --data D:\path\to\dataset.yaml \
  --long-epochs 200

# Or specify model via config file
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml
```

3.2 Suggestion JSON or direct command runs:

```powershell
# Option A: run suggestion JSON
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --suggest suggestions/suggestion_trial_0001.json \
  --data D:\path\to\dataset.yaml \
  --short-epochs 15

# Option B: invoke the training wrapper with explicit HPs
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/run_yolo_train.py \
  --data D:\path\to\dataset.yaml \
  --epochs 200 --imgsz 640 --batch 16 --lr 0.001 \
  --mosaic --name my_run --workers 0 --patience 100 \
  --topk 20 --tal-beta 2.5 --crazing-boost 2.0 --crazing-class-index 0
```

Suggestion JSON example:

```json
{
  "id": "trial-0001",
  "name": "my-experiment",
  "budget": {"epochs": 15},
  "hp": {
    "lr": 0.004,
    "batch": 16,
    "imgsz": 640,
    "mosaic": false,
    "topk": 20,
    "tal-beta": 3.0,
    "crazing-boost": 2.0
  }
}
```

3.3 Manually run Phase1 → Phase2 → Phase3 step by step:

```powershell
# Phase1 only (baseline)
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 1 --short-epochs 50 --keep-k 2

# Phase2 (reads Phase1 kept models from results.jsonl)
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 2 --short-epochs 50 --phase2-rounds 12

# Phase3: long-run validation of Phase2 champions
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 3 --long-epochs 200
```

Note: Phase2 reads `runs/autoresearch/results.jsonl` to determine Phase1-kept models; Phase3 uses Phase2 champions. Use the same `--out` path across phases.

3.4 Full automatic AutoResearch (Phase1–Phase4 one-shot):

```powershell
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --auto \
  --short-epochs 50 --long-epochs 200 \
  --keep-k 2 --phase2-rounds 12 --phase4-rounds 10 \
  --use-llm-phase4 \
  --llm-endpoint https://api.deepseek.com \
  --llm-api-key sk-xxxxxxxx \
  --llm-model deepseek-v4-flash \
  --phase4-iterations 3

# Skip phases selectively
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data ... --config ... \
  --phases "1,2,3"  # run only Phase1-3, skip Phase4
```

3.5 LLM-driven single-file code patch (examples):

```powershell
# DeepSeek
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/llm_patch_agent.py \
  --api-key $KEY --api-endpoint https://api.deepseek.com \
  --model deepseek-v4-flash \
  --target scripts/run_yolo_train.py \
  --data D:\path\to\dataset.yaml \
  --short-epochs 15 --apply

# OpenAI
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/llm_patch_agent.py \
  --api-key $KEY \
  --model gpt-4o-mini \
  --target scripts/run_yolo_train.py \
  --data D:\path\to\dataset.yaml \
  --short-epochs 15 --apply

# Anthropic Claude
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/llm_patch_agent.py \
  --api-key $KEY --api-endpoint https://api.anthropic.com \
  --model claude-3-5-sonnet \
  --target scripts/run_yolo_train.py \
  --data D:\path\to\dataset.yaml \
  --short-epochs 15 --apply
```

LLM patch workflow: LLM returns a full-file patch → AST & `py_compile` validation → `smoke` (1 epoch) → short-run → mAP comparison → Keep/Discard.

3.6 Manual progression after a phase completes:

```powershell
# Inspect Phase1-kept models
cat runs/autoresearch/results.jsonl | findstr "phase1"

# Start Phase2 manually (reads same results.jsonl)
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml --phase 2 --phase2-rounds 20

# After Phase2 finishes, run Phase3
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml --phase 3 --long-epochs 200

# Then run Phase4 with LLM guidance
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml --phase 4 --phase4-rounds 15 \
  --use-llm-phase4 --llm-endpoint https://api.deepseek.com \
  --llm-api-key sk-xxx --llm-model deepseek-v4-flash
```

## 4. `autoresearch_config.yaml` and Candidate Model Selection

Candidate model selection rules:

1. **Explicit in config (preferred)**: list YAMLs under `candidates` in `configs/autoresearch_config.yaml`.
2. **Auto-scan**: if `candidates` omitted, scan `yolov13/ultralytics/cfg/models/` and select YAMLs matching `task_type`.

Scale expansion rules:

- Files without explicit scale markers (e.g., `yolov13_nmsfree.yaml`) are expanded from family YAML `scales` into multiple scale candidates (n/s).
- Files with explicit scale markers (e.g., `yolov13n_nmsfree.yaml`) are treated as single-scale candidates.

Example config excerpt:

```yaml
# configs/autoresearch_config.yaml
mode: experiment
data: D:/path/to/dataset.yaml
task_type: detect

candidates:
  - yolov13/ultralytics/cfg/models/v13/yolov13_nmsfree.yaml
  - yolov13/ultralytics/cfg/models/v13/yolov13.yaml
  - yolov13/ultralytics/cfg/models/v13/yolov13_v10detect.yaml

short_epochs: 50
long_epochs: 200
phase2_rounds: 12
phase4_rounds: 30
keep_k: 2

use_llm_phase4: true
llm_api_key: sk-xxxxxx
llm_endpoint: https://api.deepseek.com
llm_model: deepseek-v4-flash

out: runs/autoresearch/results.jsonl
```

CLI args override config file values.

## 5. Experiment Results Lookup

After an experiment completes, all outputs are centralized under `runs/`. Here's where to find everything:

### 5.1 Experiment Ledger — `runs/autoresearch/results.jsonl`

One JSON record per line, ordered chronologically (Phase1–4, single-model, suggestion mode). Each record contains:

- `id` — run name (e.g. `phase1-yolov13n-1778511336`)
- `model` — path to model YAML
- `hp` — hyperparameter combination used
- `elapsed_sec` — training wall-time in seconds
- `weights` — path to `best.pt`
- `metrics` — mAP50 / mAP50_95 evaluation metrics
- `suggestion` (Phase4 only) — the raw LLM proposal

### 5.2 Champion Summary — `runs/autoresearch/summary.json`

Auto-generated at the end of each experiment run. Structured by phase:

| Field | Meaning |
|-------|---------|
| `phase1_kept` | Top-K models retained from Phase1 (model + HP + mAP + weight path) |
| `phase2_champions` | Best HP combination per model from Phase2 |
| `phase3_results` | Long-run validation results from Phase3 |
| `phase4_results` / `phase4_best` | All Phase4 trials + best single trial |
| `overall_best` | **Global optimum**: model path, HP, mAP, and weight file path |

### 5.3 Model Weights — `runs/detect/{run_name}/weights/`

```
runs/detect/phase3-yolov13n-1778512218/weights/
├── best.pt      ← best validation mAP checkpoint
└── last.pt      ← last epoch checkpoint
```

The `overall_best.weights` field in `summary.json` points directly to the global-best `best.pt`.

### 5.4 Training Artifacts — `runs/detect/{run_name}/`

Ultralytics auto-generates these per-run:

| File | Purpose |
|------|---------|
| `results.csv` / `results.png` | Training loss and mAP curves |
| `confusion_matrix.png` | Confusion matrix |
| `val_batch*_pred.jpg` | Validation prediction visualizations |
| `train_batch*.jpg` | Training batch samples |
| `args.yaml` | Training parameter snapshot |

### 5.5 Quick Lookup Commands

```powershell
# View full experiment ledger
Get-Content runs\autoresearch\results.jsonl

# View champion summary (pretty-printed)
Get-Content runs\autoresearch\summary.json | ConvertFrom-Json | ConvertTo-Json -Depth 5

# View only the overall best
Get-Content runs\autoresearch\summary.json | ConvertFrom-Json | Select-Object -ExpandProperty overall_best

# View Phase4 LLM suggestions
Get-Content runs\autoresearch\results.jsonl | ForEach-Object {
    $o = $_ | ConvertFrom-Json
    if ($o.id -like 'phase4*') { Write-Host "$($o.id): $($o.suggestion | ConvertTo-Json -Compress)" }
}
```

## 6. Supported LLM APIs

`lib/llm.py` auto-detects endpoint/model names and adapts authorization; you typically only need to provide `llm_endpoint`, `llm_model`, and `llm_api_key`.

Provider detection summary:

- DeepSeek: endpoint contains `api.deepseek.com` → `Authorization: Bearer {key}`; uses `/v1/chat/completions` path.
- OpenAI: OpenAI-compatible endpoints → `Authorization: Bearer {key}`.
- Ali Tongyi (通义千问): OpenAI-compatible endpoint; `Authorization: Bearer {key}`.
- Moonshot / SiliconFlow: OpenAI-compatible endpoints.
- Anthropic (Claude): endpoint/model indicates Claude → uses `x-api-key: {key}` and the Anthropic Messages API.

Example LLM configs are shown in `configs/autoresearch_config.yaml`.

## 7. Acknowledgements

This project is inspired by and built on ideas and code from two open-source projects:

- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** — the original prototype for autonomous LLM-driven experiments. Core concepts like Phase1–Phase4, Keep/Discard logic, and `program.md` originate here.
- **[iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)** — community YOLOv13 implementation. Training code, TAL assigner, and Focused-TAL in this repository leverage that implementation.

Thanks to the authors for their open-source contributions.
