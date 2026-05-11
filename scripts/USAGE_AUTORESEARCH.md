Usage and notes for orchestrator and patch_runner

1) Preconditions
- Activate the project's virtualenv and install dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

- Ensure your working tree is clean (no unstaged/uncommitted changes). `patch_runner.py` will abort otherwise.

2) Run baseline / full flow with `orchestrator.py`

Dry-run (no training executed):

```bash
python scripts/orchestrator.py --data D:\path\to\dataset.yaml --dry-run
```

Run full Phase1..Phase4 (will execute training jobs, may take long):

```bash
python scripts/orchestrator.py --data D:\path\to\dataset.yaml
```

Run only Phase1 (baseline for models):

```bash
python scripts/orchestrator.py --data D:\path\to\dataset.yaml --phase 1
```

3) Apply code patches safely with `patch_runner.py`

- Create a patch file (git diff or .patch) that modifies only allowed files (see `program.md`).
- Ensure working tree is clean.

Example:

```powershell
python scripts/patch_runner.py --patch suggestions/my_change.patch --data D:\path\to\dataset.yaml --short-epochs 15
```

What `patch_runner.py` does:
- Creates a temporary branch `autoresearch/patch-<ts>`
- Applies patch (with `git apply`)
- Runs a smoke test (1 epoch). If smoke fails, roll back to previous commit.
- Runs a short run (`--short-epochs`) and evaluate produced weights.
- If metric improves over recorded best in `runs/autoresearch/results.jsonl`, commit the patch on the new branch; otherwise rollback and delete the branch.

4) Notes and cautions
- These scripts execute real training jobs. Run them on a GPU instance with sufficient VRAM and be prepared for long execution times.
- `orchestrator.py` currently uses a simple random-search skeleton for Phase2; extend `phase2()` to use your preferred search scheduler.
- `patch_runner.py` requires a clean git state. It will attempt to reset to the previous commit on failures, but please review.

5) Next steps (suggested)
- Integrate with an LLM agent that emits `type: "hp"` JSON and `type: "code"` patches; use `patch_runner.py` to validate code patches.
- Add centralized logging and per-run timeout controls for robust large-scale runs.
