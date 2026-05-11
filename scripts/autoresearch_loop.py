#!/usr/bin/env python3
"""AutoResearch orchestrator.

Supports three usage modes:

1) Single-model training (no experiment):
   --mode train --model path/to/model.yaml --data data.yaml --long-epochs 200

2) Suggestion-based single run:
   --suggest suggestion.json --data data.yaml --short-epochs 15

3) Auto experiment (Phase1-4):
   --auto --data data.yaml --config configs/autoresearch_config.yaml
   --phases "1,2" --use-llm-phase4 --phase4-iterations 5
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from lib import models as models_lib
from lib import training as training_lib
from lib.llm import LLMClient, LLMConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """Recursively convert numpy/torch types to JSON-safe Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    except ImportError:
        pass
    return str(obj)


def append_record(path: str, rec: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(_sanitize(rec), ensure_ascii=False) + '\n')


def _now_ts():
    return int(time.time())


# ---------------------------------------------------------------------------
# single-model training (--mode train)
# ---------------------------------------------------------------------------

def run_single_model(data: str, model: str, epochs: int, out: str):
    display = Path(model).stem
    name = f'train-{display}-{_now_ts()}'
    hp = {'model': model}
    try:
        elapsed = training_lib.run_training(name, data, epochs, hp=hp)
    except Exception as e:
        append_record(out, {'id': name, 'status': 'failed', 'error': str(e)})
        print(f'Training failed: {e}')
        return
    weights = training_lib.find_best_weights(name)
    metrics = training_lib.evaluate(weights, data, name=f'{name}-val') if weights else {}
    rec = {'id': name, 'model': model, 'hp': hp, 'elapsed_sec': elapsed,
           'weights': weights, 'metrics': metrics, 'timestamp': _now_ts()}
    append_record(out, rec)
    print('Single training result:', rec)


# ---------------------------------------------------------------------------
# suggestion mode
# ---------------------------------------------------------------------------

def run_suggestion(suggest_json: str, data: str, short_epochs: int, out: str,
                   ignore_hp: bool = False):
    with open(suggest_json, 'r', encoding='utf-8') as f:
        sug = json.load(f)
    uid = sug.get('id', f'trial-{_now_ts()}')
    name = sug.get('name', uid)
    hp = {} if ignore_hp else sug.get('hp', {}) or {}
    budget = sug.get('budget', {}) or {}
    epochs = int(budget.get('epochs') or short_epochs)

    elapsed = training_lib.run_training(name, data, epochs, hp=hp)
    weights = training_lib.find_best_weights(name)
    metrics = training_lib.evaluate(weights, data, name=f'{name}-val') if weights else {}
    rec = {'id': uid, 'name': name, 'hp': hp, 'elapsed_sec': elapsed,
           'weights': weights, 'metrics': metrics}
    append_record(out, rec)
    print('Recorded suggestion result:', rec)


# ---------------------------------------------------------------------------
# Phase 1: baseline
# ---------------------------------------------------------------------------

def run_phase1(data: str, short_epochs: int, out: str, keep_k: int, task_type: str,
               candidates: list[str] | None = None):
    if candidates:
        cands = models_lib.expand_from_config(candidates, task_type)
    else:
        cands = models_lib.discover_models(task_type)
    cands = models_lib.sort_models(cands)
    results = []
    for entry in cands:
        model = entry['model']
        display = entry.get('display') or Path(model).stem
        name = f'phase1-{display}-{_now_ts()}'
        hp = {'model': model}
        if entry.get('scale'):
            hp['scale'] = entry['scale']
        try:
            elapsed = training_lib.run_training(name, data, short_epochs, hp=hp)
        except Exception as e:
            append_record(out, {'id': name, 'status': 'failed', 'error': str(e)})
            continue
        w = training_lib.find_best_weights(name)
        metrics = training_lib.evaluate(w, data, name=f'{name}-val') if w else {}
        rec = {'id': name, 'model': model, 'display': display, 'hp': hp,
               'elapsed_sec': elapsed, 'weights': w, 'metrics': metrics}
        append_record(out, rec)
        results.append(rec)

    key = lambda r: float((r.get('metrics') or {}).get('mAP50') or 0.0)
    results_sorted = sorted(results, key=key, reverse=True)
    kept = results_sorted[:keep_k]
    print('Phase1 kept:', [k['model'] for k in kept])
    return kept, results_sorted


# ---------------------------------------------------------------------------
# Phase 2: hyperparameter search
# ---------------------------------------------------------------------------

def run_phase2(data: str, kept_models: list, rounds: int, short_epochs: int, out: str):
    lr_c = [0.0005, 0.001, 0.004, 0.01]
    batch_c = [8, 16, 32]
    deg_c = [0, 3, 5]
    results = []
    for entry in kept_models:
        model = entry['model'] if isinstance(entry, dict) else entry
        display = entry.get('display') if isinstance(entry, dict) else Path(model).stem
        for i in range(rounds):
            hp = {'model': model, 'lr': random.choice(lr_c),
                  'batch': random.choice(batch_c), 'degrees': random.choice(deg_c),
                  'mosaic': random.choice([False, True])}
            name = f'phase2-{display}-r{i}-{_now_ts()}'
            try:
                elapsed = training_lib.run_training(name, data, short_epochs, hp=hp)
            except Exception as e:
                append_record(out, {'id': name, 'status': 'failed', 'error': str(e)})
                continue
            w = training_lib.find_best_weights(name)
            metrics = training_lib.evaluate(w, data, name=f'{name}-val') if w else {}
            rec = {'id': name, 'model': model, 'display': display, 'hp': hp,
                   'elapsed_sec': elapsed, 'weights': w, 'metrics': metrics}
            append_record(out, rec)
            results.append(rec)

    champions = {}
    for r in results:
        m = r['model']
        s = float((r.get('metrics') or {}).get('mAP50') or 0.0)
        if m not in champions or s > float((champions[m].get('metrics') or {}).get('mAP50') or 0.0):
            champions[m] = r
    print('Phase2 champions:', {k: v['id'] for k, v in champions.items()})
    return champions


# ---------------------------------------------------------------------------
# Phase 3: long-run validation
# ---------------------------------------------------------------------------

def run_phase3(data: str, champions: dict, long_epochs: int, out: str):
    results = []
    for m, rec in champions.items():
        base_hp = rec.get('hp', {})
        display = rec.get('display') or Path(m).stem
        name = f'phase3-{display}-{_now_ts()}'
        try:
            elapsed = training_lib.run_training(name, data, long_epochs, hp=base_hp)
        except Exception as e:
            append_record(out, {'id': name, 'status': 'failed', 'error': str(e)})
            continue
        w = training_lib.find_best_weights(name)
        metrics = training_lib.evaluate(w, data, name=f'{name}-val') if w else {}
        rec = {'id': name, 'model': m, 'display': display, 'hp': base_hp,
               'elapsed_sec': elapsed, 'weights': w, 'metrics': metrics}
        append_record(out, rec)
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Phase 4: structure & loss exploration (LLM-driven)
# ---------------------------------------------------------------------------

def run_phase4(data: str, target_model: dict, rounds: int, short_epochs: int, out: str,
               use_llm: bool = False, llm_cfg: dict | None = None, iterations: int = 3):
    results = []
    for it in range(iterations):
        print(f'Phase4 LLM iteration {it+1}/{iterations}')
        suggestions = []

        if use_llm and llm_cfg:
            client = LLMClient(LLMConfig(**llm_cfg))
            prompt = ('Provide up to 6 JSON suggestions: '
                      '[{"type":"hp","hp":{...}}]')
            resp = client.chat('You are an ML engineer. Output JSON.', prompt)
            if resp:
                try:
                    j = json.loads(resp)
                    suggestions = j.get('suggestions', [])
                except Exception:
                    import re
                    m = re.search(r'\{[\s\S]*\}', resp)
                    if m:
                        try:
                            j = json.loads(m.group(0))
                            suggestions = j.get('suggestions', [])
                        except Exception:
                            pass

        if not suggestions:
            topk_c = [16, 20, 22, 24]
            beta_c = [2.3, 2.5, 2.8, 3.0]
            boost_c = [1.0, 1.5, 2.0, 2.2]
            for _ in range(min(4, rounds)):
                suggestions.append({'type': 'hp', 'hp': {
                    'topk': random.choice(topk_c),
                    'tal-beta': random.choice(beta_c),
                    'crazing-boost': random.choice(boost_c),
                }})

        for i, s in enumerate(suggestions[:rounds]):
            print(f'  Suggestion [{i+1}/{len(suggestions[:rounds])}]: {json.dumps(s, ensure_ascii=False)}')
            if s.get('type') == 'hp':
                hp = {'model': target_model['model']}
                hp.update(s.get('hp', {}))
                name = f'phase4-{it+1}-{i}-{_now_ts()}'
                try:
                    elapsed = training_lib.run_training(name, data, short_epochs, hp=hp)
                except Exception as e:
                    append_record(out, {'id': name, 'status': 'failed', 'error': str(e)})
                    continue
                w = training_lib.find_best_weights(name)
                metrics = training_lib.evaluate(w, data, name=f'{name}-val') if w else {}
                rec = {'id': name, 'model': target_model['model'], 'hp': hp,
                       'elapsed_sec': elapsed, 'weights': w, 'metrics': metrics,
                       'suggestion': s}
                append_record(out, rec)
                results.append(rec)
            elif s.get('type') == 'struct':
                print('Struct suggestion (manual apply):', s.get('replace'))

    return results


# ---------------------------------------------------------------------------
# summary writer
# ---------------------------------------------------------------------------

def write_summary(out: str, kept: list, champions: dict,
                  phase3_results: list | None = None,
                  phase4_results: list | None = None):
    """Save a structured champion summary alongside results.jsonl."""
    summary_path = str(Path(out).parent / 'summary.json')
    payload = {}
    if kept:
        payload['phase1_kept'] = [_sanitize(k) for k in kept]
    if champions:
        payload['phase2_champions'] = {str(k): _sanitize(v) for k, v in champions.items()}
    if phase3_results:
        payload['phase3_results'] = [_sanitize(r) for r in phase3_results]
    if phase4_results:
        best_p4 = max(phase4_results,
                      key=lambda r: float((r.get('metrics') or {}).get('mAP50') or 0.0),
                      default=None)
        payload['phase4_results'] = [_sanitize(r) for r in phase4_results]
        if best_p4:
            payload['phase4_best'] = _sanitize(best_p4)

    # overall best: prefer Phase3 (long-run) > Phase4 best > champion
    best_overall = None
    if phase3_results:
        best_overall = max(phase3_results,
                           key=lambda r: float((r.get('metrics') or {}).get('mAP50') or 0.0),
                           default=None)
    elif phase4_results:
        best_overall = max(phase4_results,
                           key=lambda r: float((r.get('metrics') or {}).get('mAP50') or 0.0),
                           default=None)
    else:
        for champ_rec in champions.values():
            s = float((champ_rec.get('metrics') or {}).get('mAP50') or 0.0)
            if best_overall is None or s > float(
                    (best_overall.get('metrics') or {}).get('mAP50') or 0.0):
                best_overall = champ_rec
    if best_overall:
        payload['overall_best'] = _sanitize(best_overall)

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'Summary saved to {summary_path}')


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # --- modes ---
    parser.add_argument('--mode', default='experiment',
                        choices=['train', 'experiment'],
                        help='"train" = single model long run; "experiment" = Phase1-4')
    parser.add_argument('--model', help='Model YAML path (required for --mode train)')
    # --- core ---
    parser.add_argument('--data', required=True)
    parser.add_argument('--short-epochs', type=int, default=15)
    parser.add_argument('--long-epochs', type=int, default=100)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--phases', help='Comma-separated phases like "1,2,3"')
    parser.add_argument('--config', help='YAML config file path')
    # --- tuning knobs ---
    parser.add_argument('--keep-k', type=int, default=2)
    parser.add_argument('--phase2-rounds', type=int, default=12)
    parser.add_argument('--phase4-rounds', type=int, default=10)
    parser.add_argument('--phase4-iterations', type=int, default=3)
    # --- LLM ---
    parser.add_argument('--use-llm-phase4', action='store_true')
    parser.add_argument('--llm-endpoint')
    parser.add_argument('--llm-api-key')
    parser.add_argument('--llm-model')
    # --- misc ---
    parser.add_argument('--out', default='runs/autoresearch/results.jsonl')
    parser.add_argument('--suggest')
    parser.add_argument('--ignore-hp', action='store_true')
    parser.add_argument('--task-type', default='detect')
    args = parser.parse_args()

    # ---- YAML config overrides (CLI wins) ----
    cfg_candidates = None
    if args.config:
        from lib.config import load_config, merge_cli_overrides
        cfg = load_config(args.config)
        merge_cli_overrides(cfg, args, [
            'mode', 'model', 'short_epochs', 'long_epochs', 'keep_k',
            'phase2_rounds', 'phase4_rounds', 'phase4_iterations',
            'use_llm_phase4', 'llm_api_key', 'llm_endpoint', 'llm_model',
            'ignore_hp', 'out', 'phases', 'task_type',
        ])
        cfg_candidates = cfg.get('candidates')

    # ---- Mode: single-model training ----
    if args.mode == 'train':
        if not args.model:
            print('Error: --model is required when --mode train')
            return
        run_single_model(args.data, args.model, args.long_epochs, args.out)
        return

    # ---- Mode: suggestion ----
    if args.suggest and not args.auto and not args.phase:
        run_suggestion(args.suggest, args.data, args.short_epochs, args.out,
                       ignore_hp=args.ignore_hp)
        return

    # ---- Mode: experiment (Phase1-4) ----
    if args.phases:
        try:
            phases = set(int(x.strip()) for x in str(args.phases).split(',') if x.strip())
        except Exception:
            print('Invalid --phases, expected "1,2,3,4"')
            return
    elif args.auto or args.phase is None:
        phases = {1, 2, 3, 4}
    else:
        phases = {args.phase}

    kept, p1_all = run_phase1(args.data, args.short_epochs, args.out,
                           args.keep_k, args.task_type,
                           candidates=cfg_candidates)

    champions = {}
    if 2 in phases:
        champions = run_phase2(args.data, kept, args.phase2_rounds,
                               args.short_epochs, args.out)

    p3_results = None
    if 3 in phases:
        p3_results = run_phase3(args.data, champions, args.long_epochs, args.out)

    p4_results = None
    if 4 in phases:
        target = kept[0] if kept else (
            list(champions.values())[0] if champions else None)
        llm_cfg = None
        if args.use_llm_phase4:
            llm_cfg = {'endpoint': args.llm_endpoint, 'api_key': args.llm_api_key,
                       'model': args.llm_model}
        if target:
            p4_results = run_phase4(args.data, target, args.phase4_rounds,
                                    args.short_epochs, args.out,
                                    use_llm=args.use_llm_phase4, llm_cfg=llm_cfg,
                                    iterations=args.phase4_iterations)

    write_summary(args.out, kept, champions,
                  phase3_results=p3_results,
                  phase4_results=p4_results)


if __name__ == '__main__':
    main()
