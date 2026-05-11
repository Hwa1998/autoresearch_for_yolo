# AutoResearch 实验说明（给 Agent 的任务书模板）

目标：在受控范围内自动化进行 YOLO 训练实验，优化验证集上 `mAP@0.5`。

一、关键约束（四要素）
- **可编辑文件**：仅允许 Agent 修改下列文件或配置（其余文件禁止修改）
  - `train_yolov13.py`（训练入口）
  - `yolov13/models/*.yaml` 或项目中用于定义模型结构的 YAML 模板（当允许结构修改时）
  - `experiments/` 下的超参数 YAML（若存在）
- **不可修改文件**（Agent 只能读取）：`prepare.py`（评测与数据准备）、`program.md`（本文件）

- **标量指标**：主目标为 `mAP@0.5`（val），越高越好。次要指标：`mAP@0.5-0.95`、各类 AP。
- **单轮预算**：每轮实验的最大预算以“时间（秒）”或“短跑 epoch 数”限制，参见下文 `budget` 字段。
- **Keep / Discard**：若本轮指标优于当前最佳（基于 `mAP@0.5`），则保存 commit 并保留修改；否则 `git reset --hard` 回滚到上次保留的 commit。

二、Agent 输出格式（JSON）

Agent 每轮必须输出 JSON（stdout 或写文件），示例：

{
  "type": "hp",                # 'hp' 或 'code'
  "id": "trial-0001",         # 唯一 id
  "budget": {"epochs": 15},   # 或 {"time_seconds": 300}
  "hp": {
    "lr": 0.004,
    "batch": 16,
    "imgsz": 640,
    "mosaic": false,
    "degrees": 5,
    "topk": 20,
    "beta": 3.0,
    "crazing_boost": 2.0
  }
}

若 `type` 为 `code`，Agent 应返回一份可应用的补丁（patch 文件路径或 git diff 文本），但必须保证补丁只修改允许编辑的文件。

三、评估规则
- 使用 `prepare.py` 中的 `evaluate(weights, data)` 返回的 `mAP@0.5` 作为主判断依据。评估过程不可被 Agent 修改。

四、实验记录与命名
- 每轮实验请在运行时使用有语义的 `--name`（例如 `trial-0001-hp_lr0.004`），以便结果目录定位。
- 实验记录应包含：commit hash（若有）、suggestion JSON、实际命令、耗时、metrics（mAP 等）。

五、安全与沙箱
- Agent 的所有代码修改必须先写入临时分支并通过快速 smoke 测试（训练 1 epoch 或运行 lint），再进入短跑流程。
- 禁止执行网络下载或删除仓库外文件的操作。

六、示例流程（简短版）
1) Agent 输出 JSON (hp)。
2) Runner 应用超参并以 `--epochs` 或 `--time-budget` 启动训练（短跑）。
3) 训练结束后 runner 调用 `prepare.py:evaluate()` 获取指标。若指标 > 当前 best，则 `git commit` 并 keep；否则 `git reset --hard` 回滚。

——
（此 `program.md` 可作为 Agent 的任务书；你可以按需扩展其中的可编辑文件清单与预算策略。）
# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). For standard autoresearch experiments using the GPT training harness use `uv run train.py`. To run the YOLOv13 object-detection experiments (the `yolov13/` folder), use the provided wrapper which enforces the same wall-clock time budget:

```
uv run train_yolov13.py
```

The wrapper will run `yolov13/train.py`, terminate it after the `TIME_BUDGET` and attempt to evaluate the latest saved checkpoint. The wrapper prints a unified summary (parseable by the agent) with a `val_map` metric.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). For YOLO experiments the second column `val_bpb` is repurposed to store the primary detection metric (use `val_map`), e.g. `0.512000`. You may also add a short note in the `description` column to indicate the model family (e.g. `yolov13`).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
