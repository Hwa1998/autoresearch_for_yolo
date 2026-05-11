# AutoResearch for YOLO — 中文说明

> 本项目基于 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 开源项目实现，
> 旨在学习 AI Agent 自主实验的工程实践，并迁移到 YOLO 目标检测场景。

> 注意：本项目在重构过程中借助了 AI 助手，项目代码中可能留下明显的 AI 处理痕迹。

## 0. 项目背景

Andrej Karpathy 在 2026 年 3 月发布了 [autoresearch](https://github.com/karpathy/autoresearch)，
核心思想是：**让 LLM Agent 在一个小但真实的训练环境中整夜自主实验**——修改代码、训练、评估、
保留或丢弃改动，循环往复。本仓库将这一理念迁移到了 YOLO 目标检测方向，并加入了自己对
LLM 补丁生成、超参搜索、模型结构探索等环节的理解与实现。

## 1. 主要功能

- **Phase1-4 自动化实验循环**：基线建立 → 超参搜索 → 长跑验证 → 结构/损失探索
- **单模型训练模式**：不做实验，直接对指定模型 YAML 进行标准长跑训练
- **Suggestion JSON 驱动**：通过 JSON 文件传入超参，执行单次训练并记录结果
- **LLM 驱动代码改造**：支持 DeepSeek / OpenAI / 通义千问 / Moonshot / 硅基流动 / Anthropic Claude，让 LLM 修改训练脚本并自动验证
- **保守补丁 + 安全应用**：基于 git 分支隔离，smoke test → short run → keep/discard
- **Model YAML 变体生成**：文本级替换 Block 名称（如 C2f→C2PSA）探索结构改进
- **Focused-TAL**：通过环境变量注入，对特定类别增强 TaskAlignedAssigner 的对齐度量
- **长期自治 Agent**：无人值守循环，周期性生成/应用补丁并记录日志

## 2. 文件结构与核心模块

### 项目根目录

| 文件 | 作用 |
|------|------|
| `train.py` | GPT 预训练脚本（nanochat 简化版），Agent 可编辑，用于 LLM 预训练实验 |
| `prepare.py` | 数据准备 + 评估接口（`evaluate_placeholder`），不可修改 |
| `train_yolov13.py` | YOLOv13 时间预算训练包装器 |
| `program.md` | 给 Agent 的任务说明书（实验约束、评估规则、Keep/Discard 逻辑） |
| `configs/autoresearch_config.yaml` | 实验配置（候选模型、Phase 参数、LLM 密钥等） |

### `lib/` — 公共库

| 模块 | 行数 | 职责 |
|------|------|------|
| `lib/config.py` | ~20 | YAML 配置加载，CLI 参数优先覆盖 |
| `lib/llm.py` | ~80 | 统一 LLM 客户端，自动适配 DeepSeek/OpenAI/Anthropic |
| `lib/training.py` | ~60 | 训练启动、权重查找、评估封装 |
| `lib/models.py` | ~130 | 模型 YAML 候选发现 + 尺度展开 + 排序 |
| `lib/patches.py` | ~200 | diff 生成、安全校验、git 分支安全应用 |

### `scripts/` — 核心脚本

| 脚本 | 作用 |
|------|------|
| `scripts/autoresearch_loop.py` | ★ **主编排器**：Phase1-4 全流程 + 单模型训练 + suggestion 模式 |
| `scripts/run_yolo_train.py` | YOLO 训练包装器，超参→CLI→环境变量→tal.py |
| `scripts/autonomous_agent.py` | 长期自治守护进程，周期性 LLM 补丁生成与应用 |
| `scripts/llm_patch_agent.py` | LLM 驱动补丁生成（调用 lib/llm → 校验 → smoke → short run） |
| `scripts/llm_patch_validator.py` | AST + py_compile 双重语法校验 |
| `scripts/agent_patch_generator.py` | 保守补丁生成（只改默认超参值） |
| `scripts/patch_runner.py` | 补丁安全应用器（git 分支 → smoke → short run → keep/discard） |
| `scripts/model_variant_generator.py` | 基于文本替换的 model YAML 变体生成 |
| `scripts/debug/` | 调试脚本归档目录 |

### `yolov13/` — YOLOv13 模型实现

| 文件 | 作用 |
|------|------|
| `yolov13/ultralytics/utils/tal.py` | TaskAlignedAssigner + Focused-TAL（环境变量注入） |
| `yolov13/ultralytics/cfg/models/v13/` | 模型 YAML 配置文件 |

### Phase1-4 各阶段详解

```
═══════════════════════════════════════════════════════════════
Phase1  基础模型筛选（短 epoch × N 个候选模型）
        yolov13n, yolov13s, yolov13n_nmsfree...
        → 按 mAP@0.5 排序，保留 top-K（--keep-k，默认 2）
───────────────────────────────────────────────────────────────
Phase2  HP 随机搜索（短 epoch × K 模型 × N 轮）
        lr、batch、mosaic、degrees 随机组合
        → 每个模型的最优 HP 组合（champion）
───────────────────────────────────────────────────────────────
Phase3  长跑确认（长 epoch × K 个 champion）
        → 最终基线 mAP@0.5 / mAP@0.5-0.95
        → 产出："yolov13n_v10detect, lr=0.001, mAP@0.5=0.87"
═══════════════════════════════════════════════════════════════
                            ↓
                  上下文喂给 LLM
                            ↓
═══════════════════════════════════════════════════════════════
        ┌──────────────────────────────────────────
        │ 第 1 轮                                  
        │   LLM 看到 Phase1-3 全部结果             
        │   输出探索计划：                         
        │   · HP 方向: topk=[20,24], beta=[2.5,3.0]
        │   · 结构方向: C2f→C2PSA 值得试,          
        │              A2C2f 扩大 receptive field  
        │              ↓                           
        │   按计划批量执行：                       
        │   · Loss search: 在推荐范围内搜索        
        │   · Struct search: 生成推荐的结构变体    
        │   · 全部短训验证                         
        ├──────────────────────────────────────────
        │ 第 2 轮                                  
        │   LLM 看到第 1 轮所有结果                
        │   "topk=22 提升明显，继续探索这方向"    
        │   "C2PSA 替换反而下降，放弃"            
        │   "建议试 tal-alpha 从 1.0 调到 1.5"     
        │              ↓                           
        │   执行新一轮探索                         
        ├──────────────────────────────────────────
        │ ...重复 3~5 轮...                        
        └──────────────────────────────────────────
                            ↓
                 选出 Phase4 最优 → 跑一次长跑确认
═══════════════════════════════════════════════════════════════
```

每一步的 Keep/Discard 逻辑：以短跑后的 mAP@0.5 为主指标，若超过当前历史最优则保留（commit），否则回滚。

### 整体流程数据流

```
autoresearch_config.yaml   program.md (Agent 任务书)
        │                        │
        ▼                        ▼
┌──────────────────────────────────────────────────────┐
│       scripts/autoresearch_loop.py                   │
│                (Orchestrator)                        │
│                                                      │
│  --mode train  →  run_single_model()                 │
│  --suggest     →  run_suggestion()                   │
│  --auto        →  Phase1 → Phase2 → Phase3 → Phase4  │
│                                                      │
│  run_training() → subprocess → run_yolo_train.py     │
│  evaluate() ────────────────→ prepare.py             │
│  append_record() ───────────→ results.jsonl          │
└──────────────┬───────────────────────────────────────┘
               │
    ┌──────────┼────────────┬──────────────┐
    ▼          ▼            ▼              ▼
 lib/training  lib/llm   lib/patches   lib/models
 (训练运行)   (LLM客户端) (补丁工具)   (模型发现)
```

## 3. 启动方式

> **环境说明**：本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境。
> 克隆仓库后，在项目根目录执行 `uv venv` 创建 `.venv`，再用 `uv pip install -r requirements.txt` 安装依赖。
>
> 所有命令示例均假设你在项目根目录下、使用 Windows PowerShell，且已激活虚拟环境。
> Windows 下运行时需要先设置 `PYTHONPATH` 确保模块导入正确：

### 3.1 直接在终端进行单模型训练

```powershell
# 使用默认超参，对指定模型 YAML 进行长跑训练（200 epochs）
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --mode train \
  --model yolov13/ultralytics/cfg/models/v13/yolov13.yaml \
  --data D:\path\to\dataset.yaml \
  --long-epochs 200

# 通过 config 文件指定模型
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml
```

### 3.2 使用 Suggestion JSON 或直接命令进行训练

```powershell
# 方式 A：使用 suggestion JSON 文件
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --suggest suggestions/suggestion_trial_0001.json \
  --data D:\path\to\dataset.yaml \
  --short-epochs 15

# 方式 B：直接调用训练脚本，显式传入所有超参
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/run_yolo_train.py \
  --data D:\path\to\dataset.yaml \
  --epochs 200 --imgsz 640 --batch 16 --lr 0.001 \
  --mosaic --name my_run --workers 0 --patience 100 \
  --topk 20 --tal-beta 2.5 --crazing-boost 2.0 --crazing-class-index 0
```

Suggestion JSON 示例格式：
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

### 3.3 手动运行 Phase1 → Phase2 → Phase3

当你只想逐步手动推进实验时：

```powershell
# Step 1: 只跑 Phase1（建立基线）
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 1 --short-epochs 50 --keep-k 2

# Step 2: 基于 Phase1 结果，手动运行 Phase2（超参搜索）
# Phase2 会从 results.jsonl 中自动读取 Phase1 保留的模型
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 2 --short-epochs 50 --phase2-rounds 12

# Step 3: 长跑验证 Phase2 冠军
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --config configs/autoresearch_config.yaml \
  --phase 3 --long-epochs 200
```

> **注意**：Phase2 会自动从 `runs/autoresearch/results.jsonl` 中读取 Phase1 的结果来确定
> 哪些模型被保留。Phase3 同样依赖 Phase2 的 champion 记录。因此务必确保使用同一个 `--out` 路径。

### 3.4 全自动 AutoResearch 实验（Phase1-4 一键运行）

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

# 也可以选择性跳过某些阶段
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data ... --config ... \
  --phases "1,2,3"    # 只跑 Phase1-3，跳过 Phase4
```

### 3.5 LLM 驱动的单次代码改造

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

流程：LLM 返回改后的完整文件 → AST/py_compile 校验 → smoke(1 epoch) → short run → mAP 对比 → Keep/Discard。

### 3.6 Phase 完成后手动推进下一阶段

```powershell
# Phase1 跑完后，查看哪些模型被保留
cat runs/autoresearch/results.jsonl | findstr "phase1"

# 手动启动 Phase2（读取同一 results.jsonl）
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --phase 2 --phase2-rounds 20

# Phase2 跑完后启动 Phase3
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --phase 3 --long-epochs 200

# 基于前三个阶段的结果启动 Phase4
$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH; .venv\Scripts\python.exe scripts/autoresearch_loop.py \
  --data D:\path\to\dataset.yaml \
  --phase 4 --phase4-rounds 15 \
  --use-llm-phase4 \
  --llm-endpoint https://api.deepseek.com \
  --llm-api-key sk-xxx --llm-model deepseek-v4-flash
```

## 4. `autoresearch_config.yaml` 与模型候选选择

### 候选模型规则

在 Phase1 选择 model YAML 时，仓库支持两种方式：

1. **Config 显式指定**（优先）：在 `autoresearch_config.yaml` 中通过 `candidates` 字段列举
2. **自动扫描**：若未指定 candidates，自动扫描 `yolov13/ultralytics/cfg/models/`，筛选匹配 `task_type` 的 YAML

### 尺度展开规则

- **不含尺度标记的文件**（如 `yolov13_nmsfree.yaml`）：从 family YAML（`yolov13.yaml`）的 `scales` 段自动展开为 n/s 两个尺度
- **含尺度标记的文件**（如 `yolov13n_nmsfree.yaml`）：判定已指定尺度，只生成 1 个候选

示例 config：
```yaml
# configs/autoresearch_config.yaml
mode: experiment
data: D:/path/to/dataset.yaml
task_type: detect

# Phase1 候选模型
candidates:
  - yolov13/ultralytics/cfg/models/v13/yolov13_nmsfree.yaml
  - yolov13/ultralytics/cfg/models/v13/yolov13.yaml
  - yolov13/ultralytics/cfg/models/v13/yolov13_v10detect.yaml

# Phase 参数
short_epochs: 50
long_epochs: 200
phase2_rounds: 12
phase4_rounds: 30
keep_k: 2

# LLM 配置
use_llm_phase4: true
llm_api_key: sk-xxxxxx
llm_endpoint: https://api.deepseek.com
llm_model: deepseek-v4-flash

# 输出
out: runs/autoresearch/results.jsonl
```

> **CLI 参数优先**：命令行传入的任何参数会覆盖 config 文件中的同名配置。

### 支持的 LLM API

`lib/llm.py` 自动检测 endpoint / model 名称并适配认证方式，无需手动区分：

| 提供商 | 触发条件 | Authorization 头 | 备注 |
|--------|----------|-------------------|------|
| **DeepSeek** | endpoint 以 `api.deepseek.com` 结尾 | `Bearer {key}` | 自动拼接 `/v1/chat/completions` |
| **OpenAI** | 默认（OpenAI 兼容 chat completions） | `Bearer {key}` | `https://api.openai.com/v1/chat/completions` |
| **通义千问（阿里）** | OpenAI 兼容格式 | `Bearer {key}` | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` |
| **Moonshot（月之暗面）** | OpenAI 兼容格式 | `Bearer {key}` | `https://api.moonshot.cn/v1/chat/completions` |
| **硅基流动（SiliconFlow）** | OpenAI 兼容格式 | `Bearer {key}` | `https://api.siliconflow.cn/v1/chat/completions` |
| **Anthropic (Claude)** | endpoint 含 `anthropic` 或 model 含 `claude` | `x-api-key: {key}` | 使用 Anthropic Messages API |

各平台 config 配置示例：

```yaml
llm_api_key: sk-xxx

# DeepSeek
llm_endpoint: https://api.deepseek.com
llm_model: deepseek-v4-flash

# 通义千问（阿里）
llm_endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
llm_model: qwen-plus

# Moonshot（月之暗面）
llm_endpoint: https://api.moonshot.cn/v1/chat/completions
llm_model: moonshot-v1-8k
```

## 5. 实验结果查看

实验完成后，所有结论和产出物集中存放在 `runs/` 目录下。以下是各产物的位置与用途：

### 5.1 实验总账 — `runs/autoresearch/results.jsonl`

每行一条 JSON，按时间顺序记录所有试验（Phase1~4、单模型训练、suggestion 模式）。包含字段：

- `id`：实验名称（如 `phase1-yolov13n-1778511336`）
- `model`：使用的模型 YAML 路径
- `hp`：该次试验的超参组合
- `elapsed_sec`：训练耗时（秒）
- `weights`：最优权重文件路径（`best.pt`）
- `metrics`：mAP50 / mAP50_95 等评估指标
- `suggestion`（仅 Phase4）：LLM 给出的原始建议

### 5.2 冠军总结 — `runs/autoresearch/summary.json`

实验结束后自动生成的结构化总结，按阶段分层：

| 字段 | 含义 |
|------|------|
| `phase1_kept` | Phase1 保留的 top-K 模型（模型路径 + HP + mAP + 权重路径） |
| `phase2_champions` | 每个模型的 Phase2 最优超参组合 |
| `phase3_results` | Phase3 长跑验证的最终结果 |
| `phase4_results` / `phase4_best` | Phase4 所有探索记录 + 单轮最优 |
| `overall_best` | **全局最优**：模型路径、超参、mAP、权重文件路径 |

### 5.3 模型权重 — `runs/detect/{run_name}/weights/`

每次训练完成后，权重保存在对应 `run_name` 目录下：

```
runs/detect/phase3-yolov13n-1778512218/weights/
├── best.pt      ← 验证集 mAP 最优权重
└── last.pt      ← 最后一个 epoch 的权重
```

其中 `overall_best` 记录的 `weights` 路径即指向全局最优的 `best.pt`。

### 5.4 训练详情 — `runs/detect/{run_name}/`

每个 run 目录还包含 Ultralytics 自动产出的训练分析文件：

| 文件 | 用途 |
|------|------|
| `results.csv` / `results.png` | 训练 loss/mAP 曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `val_batch*_pred.jpg` | 验证集预测可视化 |
| `train_batch*.jpg` | 训练集样本可视化 |
| `args.yaml` | 训练参数快照 |

### 5.5 快速查看命令

```powershell
# 查看实验总账
Get-Content runs\autoresearch\results.jsonl

# 查看冠军总结（格式化输出）
Get-Content runs\autoresearch\summary.json | ConvertFrom-Json | ConvertTo-Json -Depth 5

# 只看全局最优
Get-Content runs\autoresearch\summary.json | ConvertFrom-Json | Select-Object -ExpandProperty overall_best

# 查看 Phase4 的 LLM 建议
Get-Content runs\autoresearch\results.jsonl | ForEach-Object {
    $o = $_ | ConvertFrom-Json
    if ($o.id -like 'phase4*') { Write-Host "$($o.id): $($o.suggestion | ConvertTo-Json -Compress)" }
}
```

## 6. 致谢

本项目基于以下两个优秀开源项目的 idea 和代码：

- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** — Andrej Karpathy 的"让 AI Agent 自己做实验"的原型项目。本项目的 Phase1-4 循环、Keep/Discard 逻辑、program.md 任务书等核心概念均来源于此。
- **[iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)** — YOLOv13 目标检测模型的社区实现。本项目中的模型训练、TAL assigner、Focused-TAL 等基于其 ultralytics 框架代码。

感谢两位作者的开源精神和对社区的无私贡献。

---
