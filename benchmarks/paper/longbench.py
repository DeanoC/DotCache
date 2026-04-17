"""LongBench benchmark (English subset): dense vs certified attention.

Mirrors benchmarks/paper/ruler.py.  Unlike RULER (which synthesises prompts
from templates), LongBench is real data from the HuggingFace `THUDM/LongBench`
dataset.  We evaluate on the 13 English subtasks:

    single-doc QA : narrativeqa, qasper, multifieldqa_en
    multi-doc QA  : hotpotqa, 2wikimqa
    summarisation : gov_report, qmsum, multi_news
    few-shot      : trec, triviaqa, samsum
    code          : lcc, repobench-p

Prompt templates, max_new_tokens, and scoring metrics are lifted verbatim
from the upstream LongBench repo (THUDM/LongBench, main branch) so our paired
dense-vs-certified numbers are directly comparable to the upstream reference
formulation.  We do NOT claim numerical parity with the upstream leaderboard —
ROUGE and fuzz implementations may differ slightly — but within a single run
the dense and certified paths use the same scorer, so Δ is meaningful.

Samples longer than the target context (ctx_tokens) are middle-truncated on
the `context` field: keep the first and last halves of `context`, drop the
middle.  This is the convention upstream LongBench uses for length control.

Run: dense and certified generate on identical prompts; paired comparison.
Critical failure = certified regresses >20 pts where dense scored well.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Upstream LongBench config (verbatim from THUDM/LongBench, main branch)
# ---------------------------------------------------------------------------

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lcc": 64,
    "repobench-p": 64,
}

SUBTASKS = list(DATASET2PROMPT.keys())
# Subtasks whose predictions are truncated to the first line before scoring.
FIRST_LINE_SUBTASKS = {"trec", "triviaqa", "samsum"}


# ---------------------------------------------------------------------------
# Metrics (adapted from THUDM/LongBench/metrics.py)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lower, strip articles, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    p_tokens = _normalize_answer(prediction).split()
    g_tokens = _normalize_answer(ground_truth).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


_ROUGE_SCORER = None


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def rouge_l_score(prediction: str, ground_truth: str, **kwargs) -> float:
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    try:
        sc = _get_rouge_scorer().score(ground_truth, prediction)
        return sc["rougeL"].fmeasure
    except Exception:
        return 0.0


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Upstream TREC scorer: check which class names appear in prediction,
    remove ones that are substrings of others (shorter label overlap), then
    score 1/N if ground_truth is in the remaining set.
    """
    all_classes = kwargs.get("all_classes") or []
    em_matches = [c for c in all_classes if c in prediction]
    # Drop matches that are strict substrings of ground_truth (e.g. "NUM" when
    # the real label is "NUMBER") — upstream's exact check mirrored here.
    em_matches = [c for c in em_matches
                  if not (c in ground_truth and c != ground_truth)]
    if ground_truth in em_matches:
        return 1.0 / len(em_matches)
    return 0.0


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Upstream code_sim: take the first non-comment/non-backtick line of the
    prediction and fuzz.ratio it against ground_truth / 100.
    """
    from rapidfuzz import fuzz
    first = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" in line or "#" in line or "//" in line:
            continue
        first = line
        break
    return fuzz.ratio(first, ground_truth) / 100.0


DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_l_score,
    "qmsum": rouge_l_score,
    "multi_news": rouge_l_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_l_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def score_sample(subtask: str, prediction: str, references: list[str],
                 all_classes: list[str] | None) -> float:
    """LongBench per-sample score: max over reference answers."""
    if subtask in FIRST_LINE_SUBTASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    metric = DATASET2METRIC[subtask]
    if not references:
        return 0.0
    best = 0.0
    for ref in references:
        s = metric(prediction, ref, all_classes=all_classes)
        if s > best:
            best = s
    return best


# ---------------------------------------------------------------------------
# Dataset + prompt construction
# ---------------------------------------------------------------------------

_LONGBENCH_SAMPLES_CACHE: dict[str, list[dict]] = {}


def _load_subtask(subtask: str) -> list[dict]:
    """Load (and cache) a LongBench subtask as a list of sample dicts.

    HF `datasets` 4.x dropped script-dataset support, and THUDM/LongBench is
    still distributed as a `data.zip` + loader script.  We sidestep that by
    downloading the zip once (huggingface_hub caches it under
    ~/.cache/huggingface/hub) and reading the per-subtask JSONL directly.
    """
    if subtask in _LONGBENCH_SAMPLES_CACHE:
        return _LONGBENCH_SAMPLES_CACHE[subtask]

    import zipfile
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    zip_path = hf_hub_download(
        "THUDM/LongBench", "data.zip", repo_type="dataset", token=token,
    )
    member = f"data/{subtask}.jsonl"
    samples: list[dict] = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    _LONGBENCH_SAMPLES_CACHE[subtask] = samples
    return samples


def _middle_truncate_context(context: str, ctx_tokens: int,
                              wrapper_tokens: int, tokenizer) -> str:
    """Truncate `context` so the full prompt fits within ctx_tokens.

    Preserves the first and last halves and drops the middle — LongBench
    convention.  `wrapper_tokens` is the token count of the template with an
    empty context (includes the input/question and preamble).
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    budget = ctx_tokens - wrapper_tokens - 32  # safety margin
    if budget <= 0 or len(ids) <= budget:
        return context
    half = budget // 2
    kept = ids[:half] + ids[-half:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def build_longbench_prompt(sample: dict, subtask: str, tokenizer,
                           ctx_tokens: int) -> tuple[str, int]:
    """Build a (possibly truncated) LongBench prompt.  Returns (prompt, seq_len)."""
    template = DATASET2PROMPT[subtask]
    context = sample.get("context", "")
    input_q = sample.get("input", "")

    wrapper = template.format(context="", input=input_q)
    wrapper_tokens = len(tokenizer.encode(wrapper, add_special_tokens=False))
    ctx_trunc = _middle_truncate_context(context, ctx_tokens,
                                         wrapper_tokens, tokenizer)
    prompt = template.format(context=ctx_trunc, input=input_q)
    seq_len = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, seq_len


# ---------------------------------------------------------------------------
# Generation paths — mirror ruler.py (dense FP16 attn, certified full pipeline)
# ---------------------------------------------------------------------------

def generate_dense(model, tokenizer, adapter, prompt: str, max_new: int,
                   device: str = "cuda") -> tuple[str, int]:
    adapter.set_mode("dense")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, temperature=1.0,
        )
    gen_ids = outputs[0, seq_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, seq_len


def generate_certified(model, tokenizer, adapter, prompt: str, max_new: int,
                       calibrated_profile=None, default_epsilon: float = 1e-4,
                       top_k_fp16_keys: int = 4,
                       concentration_threshold: float = 0.02,
                       device: str = "cuda") -> tuple[str, int]:
    from dotcache.integrations.llama import (
        _ensure_certified_imports, CertifiedAttentionState,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    adapter.set_mode("dense")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del outputs

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered_caches = create_tiered_cache_from_model(
        past_kv, layer_ids, max_new_tokens=max_new + 8,
    )
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    if calibrated_profile is not None:
        layer_epsilons = calibrated_profile.get_layer_epsilons_min(seq_len)
    else:
        layer_epsilons = {}

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered_caches,
        layer_epsilons=layer_epsilons,
        default_epsilon=default_epsilon,
        collect_stats=False,
        append_kv=True,
        top_k_fp16_keys=top_k_fp16_keys,
        concentration_threshold=concentration_threshold,
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token
    gen_ids = [first_token.item()]
    for _ in range(max_new - 1):
        with torch.inference_mode():
            out = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        tid = out.logits[:, -1, :].argmax(dim=-1).item()
        gen_ids.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        current_input = torch.tensor([[tid]], dtype=torch.long, device=device)
        cache_position = cache_position + 1

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    adapter.certified_state = None
    adapter.set_mode("dense")
    gc.collect()
    torch.cuda.empty_cache()
    return text, seq_len


# ---------------------------------------------------------------------------
# Top-level sweep
# ---------------------------------------------------------------------------

def _sample_indices(n_total: int, num_samples: int, subtask: str,
                    ctx_tokens: int, seed_base: int) -> list[int]:
    """Deterministic per (subtask, ctx) sample selection.

    If num_samples >= n_total, use all.  Otherwise use a stable md5-derived
    permutation of range(n_total) and take the first num_samples.  This lets
    us reproduce exact sample sets across machines and runs.
    """
    if num_samples >= n_total:
        return list(range(n_total))
    key = f"{subtask}|{ctx_tokens}|{seed_base}".encode()
    seed = int(hashlib.md5(key).hexdigest()[:8], 16)
    import random as _r
    rng = _r.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    return sorted(idx[:num_samples])


def run_longbench(
    model, tokenizer, adapter,
    subtasks: list[str], contexts: list[int], num_samples: int,
    calibrated_profile=None, default_epsilon: float = 1e-4,
    top_k_fp16_keys: int = 4, concentration_threshold: float = 0.02,
    seed_base: int = 20260417, device: str = "cuda",
) -> dict:
    # Pre-load all subtasks once (HF datasets cache to ~/.cache/huggingface)
    print("Loading LongBench subtasks...")
    subtask_data = {}
    for st in subtasks:
        ds = _load_subtask(st)
        subtask_data[st] = ds
        print(f"  {st:<18} {len(ds)} samples")

    results = []
    total = sum(min(num_samples, len(subtask_data[st]))
                for st in subtasks) * len(contexts) * 2
    done = 0

    for ctx_len in contexts:
        for subtask in subtasks:
            ds = subtask_data[subtask]
            max_new = DATASET2MAXLEN[subtask]
            indices = _sample_indices(len(ds), num_samples, subtask,
                                      ctx_len, seed_base)

            for sidx in indices:
                sample = ds[sidx]
                refs = list(sample.get("answers") or [])
                all_classes = sample.get("all_classes") or None
                prompt, seq_expected = build_longbench_prompt(
                    sample, subtask, tokenizer, ctx_len,
                )

                dense_text, seq_dense = generate_dense(
                    model, tokenizer, adapter, prompt, max_new, device=device,
                )
                dense_score = score_sample(subtask, dense_text, refs, all_classes)
                done += 1
                print(f"  [{done}/{total}] dense     {subtask:<18} "
                      f"{ctx_len//1024}K s={sidx} seq={seq_dense} "
                      f"-> {dense_score:.2f}")

                cert_text, seq_cert = generate_certified(
                    model, tokenizer, adapter, prompt, max_new,
                    calibrated_profile=calibrated_profile,
                    default_epsilon=default_epsilon,
                    top_k_fp16_keys=top_k_fp16_keys,
                    concentration_threshold=concentration_threshold,
                    device=device,
                )
                cert_score = score_sample(subtask, cert_text, refs, all_classes)
                done += 1
                # Critical = cert dropped >0.2 below dense on a sample dense
                # scored well (>=0.5) — mirror NIAH's "regression gate".
                crit = (dense_score >= 0.5
                        and cert_score < dense_score - 0.2)
                flag = "  *** CRITICAL ***" if crit else ""
                print(f"  [{done}/{total}] certified {subtask:<18} "
                      f"{ctx_len//1024}K s={sidx} seq={seq_cert} "
                      f"-> {cert_score:.2f}{flag}")

                results.append({
                    "subtask": subtask, "ctx_tokens": ctx_len,
                    "sample_idx": sidx, "seq_len": seq_dense,
                    "refs": refs, "all_classes": all_classes,
                    "dense_score": dense_score, "dense_gen": dense_text[:400],
                    "cert_score": cert_score, "cert_gen": cert_text[:400],
                    "critical": bool(crit),
                })

    # Aggregate per (subtask, ctx)
    by_task_ctx = {}
    for r in results:
        key = (r["subtask"], r["ctx_tokens"])
        agg = by_task_ctx.setdefault(key, {"dense": [], "cert": [], "crit": 0})
        agg["dense"].append(r["dense_score"])
        agg["cert"].append(r["cert_score"])
        if r["critical"]:
            agg["crit"] += 1

    summary = {}
    for (subtask, ctx), agg in by_task_ctx.items():
        summary[f"{subtask}_{ctx//1024}K"] = {
            "dense_mean": sum(agg["dense"]) / len(agg["dense"]),
            "cert_mean": sum(agg["cert"]) / len(agg["cert"]),
            "critical": agg["crit"],
            "n": len(agg["dense"]),
        }
    return {"results": results, "summary": summary}


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig
    from dotcache.calibration.calibrated_profile import CalibratedProfile

    parser = argparse.ArgumentParser(description="LongBench (EN subset): dense vs certified")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--subtasks", nargs="+", default=SUBTASKS)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output", default="benchmarks/results/longbench.json")
    parser.add_argument("--default-epsilon", type=float, default=1e-4)
    parser.add_argument("--top-k-fp16", type=int, default=4)
    parser.add_argument("--concentration-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260417)
    args = parser.parse_args()

    for st in args.subtasks:
        if st not in DATASET2PROMPT:
            raise SystemExit(f"Unknown subtask: {st} (valid: {SUBTASKS})")

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_config, device_map="auto", token=token,
    )
    model.eval()
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    config = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, config)

    profile = None
    if args.profile:
        profile = CalibratedProfile.load(args.profile)
        print(f"Loaded profile: {profile.summary()[:200]}")

    print(f"\nLongBench: subtasks={args.subtasks}, "
          f"contexts={[c//1024 for c in args.contexts]}K, "
          f"n={args.num_samples}, eps_default={args.default_epsilon}, "
          f"top_k_fp16={args.top_k_fp16}, conc_thr={args.concentration_threshold}")

    t0 = time.perf_counter()
    out = run_longbench(
        model, tokenizer, adapter,
        subtasks=args.subtasks, contexts=args.contexts,
        num_samples=args.num_samples, calibrated_profile=profile,
        default_epsilon=args.default_epsilon,
        top_k_fp16_keys=args.top_k_fp16,
        concentration_threshold=args.concentration_threshold,
        seed_base=args.seed,
    )
    wall = time.perf_counter() - t0

    print(f"\n{'='*64}")
    print(f"LongBench (EN) results ({wall/60:.1f} min)")
    print(f"{'='*64}")
    print(f"{'task_ctx':<26} {'dense':>8} {'cert':>8} {'Δ':>8} {'crit':>6}")
    for key, s in sorted(out["summary"].items()):
        delta = s["cert_mean"] - s["dense_mean"]
        print(f"{key:<26} {s['dense_mean']:>8.3f} {s['cert_mean']:>8.3f} "
              f"{delta:>+8.3f} {s['critical']:>6}")

    dmean = [r["dense_score"] for r in out["results"]]
    cmean = [r["cert_score"] for r in out["results"]]
    critical = sum(1 for r in out["results"] if r["critical"])
    overall_d = sum(dmean) / len(dmean) if dmean else 0.0
    overall_c = sum(cmean) / len(cmean) if cmean else 0.0
    print(f"\nOverall: dense={overall_d:.3f} cert={overall_c:.3f} "
          f"Δ={overall_c-overall_d:+.3f} critical={critical}/{len(dmean)}")

    payload = {
        "benchmark": "longbench_en_subset",
        "model": args.model,
        "subtasks": args.subtasks,
        "contexts": args.contexts,
        "num_samples": args.num_samples,
        "default_epsilon": args.default_epsilon,
        "top_k_fp16": args.top_k_fp16,
        "concentration_threshold": args.concentration_threshold,
        "seed": args.seed,
        "wall_minutes": wall / 60.0,
        "overall_dense": overall_d,
        "overall_cert": overall_c,
        "critical_failures": critical,
        "summary": out["summary"],
        "results": out["results"],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
