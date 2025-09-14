#!/usr/bin/env python3
"""
MultiLoKo evaluation runner for comparing Baseline vs Nospher prompts.

What this script does
- Loads MultiLoKo dev split for one or more languages from the decrypted
  `benchmark_data/<lang>/dev.jsonl` and `knowledge_fewshot.jsonl` files.
- Builds prompts with/without your Nospher system prompt.
- Calls an LLM (OpenAI by default) asynchronously.
- Writes predictions to JSONL and CSV.
- Computes simple Exact Match (EM) and character-level F1 locally.

Optional: If you want to also score with the official script, you can
export a file that mirrors the examples schema and call `eval.py` 
from the official repo.

Usage examples
--------------
# 1) Baseline (no Nospher), Japanese (ja), first 50 examples
python multiloko_nospher_eval.py \
  --dataset-root ./benchmark_data \
  --langs ja \
  --max-samples 50 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --out-dir runs/baseline

# 2) Nospher prompt ON (--nospher-prompt-fileを指定することで自動的にnospherモード)
python multiloko_nospher_eval.py \
  --dataset-root ./benchmark_data \
  --langs ja \
  --max-samples 50 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --nospher-prompt-file nospher.txt \
  --out-dir runs/nospher

# 3) Compare two run folders (prints EM diff)
python multiloko_nospher_eval.py --compare runs/baseline runs/nospher

Notes
- Set OPENAI_API_KEY in your environment when using --provider openai
- The dataset must be extracted as documented in the MultiLoKo README
  (decrypt the .enc using password "multiloko").
- This script favors correctness and clarity over micro-optimizations.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
load_dotenv()

# --------------------------- Utilities ---------------------------------

JA = {"ja", "jp", "jpn"}


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def normalize_text(s: str, lang: str) -> str:
    """Language-agnostic normalization for short-answer matching.
    - Unicode NFKC normalization
    - Lowercase (for scripts with case)
    - Trim & collapse whitespace
    - Remove most punctuation
    - Normalize digits to ASCII
    - Strip surrounding quotes
    """
    if s is None:
        return ""
    s = nfkc(s)
    # Lowercase for languages with case
    if lang not in JA:
        s = s.lower()
    # Remove most punctuation (keep intra-word hyphens/dots)
    s = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F'\"“”‘’`´，、。！？；：·•…（）〔〕【】《》〈〉［］｛｝()\[\]{}，,，.:;!?]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Strip surrounding quotes again
    s = s.strip('"\'')
    return s


def char_f1(pred: str, gold: str) -> float:
    """Character-level F1 for languages without whitespace tokenization.
    (Simple symmetric set overlap over characters.)"""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p = list(pred)
    g = list(gold)
    common = 0
    g_counts: Dict[str, int] = {}
    for ch in g:
        g_counts[ch] = g_counts.get(ch, 0) + 1
    for ch in p:
        if g_counts.get(ch, 0) > 0:
            common += 1
            g_counts[ch] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


# ---------------------- Data structures --------------------------------

@dataclass
class QAExample:
    lang: str
    idx: int  # index within the dev.jsonl for that language
    question: str
    context: Optional[str]
    targets: List[str]  # list of acceptable short answers
    output_type: Optional[str] = None


# ------------------------- IO ------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_multiloko_dev(dataset_root: Path, langs: Sequence[str], max_samples: Optional[int]) -> List[QAExample]:
    """Load dev.jsonl for each requested language.
    Expects directory structure: benchmark_data/<lang>/dev.jsonl
    Also attempts to read `output_type` if present.
    """
    examples: List[QAExample] = []
    for lang in langs:
        lang_dir = dataset_root / lang
        dev_path = lang_dir / "dev.jsonl"
        if not dev_path.exists():
            raise FileNotFoundError(f"Missing {dev_path}. Ensure dataset is decrypted to {dataset_root}.")
        raw = load_jsonl(dev_path)
        if max_samples:
            raw = raw[: max_samples]
        for i, r in enumerate(raw):
            examples.append(
                QAExample(
                    lang=lang,
                    idx=i,
                    question=r.get("question", "").strip(),
                    context=(r.get("text") or None),
                    targets=[t for t in r.get("targets", []) if isinstance(t, str)],
                    output_type=r.get("output_type"),
                )
            )
    return examples


def load_fewshots(dataset_root: Path, lang: str, k: int = 5) -> List[Tuple[str, str]]:
    """Returns list of (question, answer) pairs from knowledge_fewshot.jsonl"""
    fpath = dataset_root / lang / "knowledge_fewshot.jsonl"
    if not fpath.exists():
        return []
    rows = load_jsonl(fpath)
    pairs: List[Tuple[str, str]] = []
    for r in rows[:k]:
        q = (r.get("question") or "").strip()
        # Prefer short target if available
        tlist = r.get("targets") or []
        ans = (tlist[0] if tlist else (r.get("target") or "")).strip()
        pairs.append((q, ans))
    return pairs


# ----------------------- Prompting -------------------------------------

DEFAULT_BASELINE_SYSTEM = (
    "You are a precise QA assistant.\n"
    "Answer with a single word or the shortest possible phrase, in the same language as the question.\n"
    "Do not include explanations or punctuation, just the answer."
)


def build_messages(
    ex: QAExample,
    fewshots: List[Tuple[str, str]],
    nospher: bool,
    nospher_system_override: Optional[str] = None,
) -> List[Dict[str, str]]:
    if nospher:
        if not nospher_system_override:
            raise ValueError("Nospher mode requires --nospher-prompt-file to be specified with a valid file")
        system = nospher_system_override + "\n" + DEFAULT_BASELINE_SYSTEM
    else:
        system = DEFAULT_BASELINE_SYSTEM

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # Add few-shot Q/A examples as user/assistant turns
    for q, a in fewshots:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    user_prompt = []
    if ex.context:
        user_prompt.append(f"[文脈]\n{ex.context}\n")
    user_prompt.append("[質問]\n" + ex.question)
    user_prompt.append("\n[出力形式]\n答えのみ。1語または短いフレーズ。余計な語・記号なし。")

    messages.append({"role": "user", "content": "\n".join(user_prompt)})
    return messages


# ----------------------- LLM Providers ---------------------------------

class LLMProvider:
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 32):
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. `pip install openai`"
            ) from e
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Use Chat Completions for broad compatibility
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()


# -------------------------- Runner -------------------------------------

@dataclass
class PredRow:
    lang: str
    idx: int
    question: str
    prediction_raw: str
    prediction_norm: str
    gold_norm_list: List[str]
    em: int
    f1: float


async def generate_for_example(
    provider: LLMProvider,
    ex: QAExample,
    fewshots: List[Tuple[str, str]],
    nospher: bool,
    nospher_system: Optional[str],
) -> PredRow:
    msgs = build_messages(ex, fewshots, nospher, nospher_system)
    out = await provider.generate(msgs)
    pred_raw = out.strip().splitlines()[0] if out else ""  # first line only
    pred_norm = normalize_text(pred_raw, ex.lang)
    gold_norms = [normalize_text(g, ex.lang) for g in ex.targets]
    em = int(pred_norm in set(gold_norms))
    best_f1 = max((char_f1(pred_norm, g) for g in gold_norms), default=0.0)
    return PredRow(
        lang=ex.lang,
        idx=ex.idx,
        question=ex.question,
        prediction_raw=pred_raw,
        prediction_norm=pred_norm,
        gold_norm_list=gold_norms,
        em=em,
        f1=best_f1,
    )


async def run_eval(
    dataset_root: Path,
    langs: Sequence[str],
    provider: LLMProvider,
    max_samples: Optional[int],
    nospher: bool,
    nospher_system: Optional[str],
    concurrency: int,
    seed: int,
) -> List[PredRow]:
    random.seed(seed)
    # Load data per language + fewshots
    all_examples: List[QAExample] = []
    fewshot_map: Dict[str, List[Tuple[str, str]]] = {}
    for lang in langs:
        fewshot_map[lang] = load_fewshots(dataset_root, lang)
    all_examples = load_multiloko_dev(dataset_root, langs, max_samples)

    sem = asyncio.Semaphore(concurrency)

    async def _task(ex: QAExample) -> PredRow:
        async with sem:
            return await generate_for_example(
                provider, ex, fewshot_map.get(ex.lang, []), nospher, nospher_system
            )

    results: List[PredRow] = []
    coros = [_task(ex) for ex in all_examples]
    for fut in asyncio.as_completed(coros):
        try:
            row = await fut
            results.append(row)
            # simple progress
            if len(results) % 10 == 0:
                print(f"done: {len(results)}/{len(all_examples)}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR during generation: {e}", file=sys.stderr)
    return results


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[PredRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "lang",
            "idx",
            "question",
            "prediction_raw",
            "prediction_norm",
            "gold_norm_list",
            "em",
            "f1",
        ])
        for r in rows:
            w.writerow([
                r.lang,
                r.idx,
                r.question,
                r.prediction_raw,
                r.prediction_norm,
                "|".join(r.gold_norm_list),
                r.em,
                f"{r.f1:.4f}",
            ])


def to_official_pred_format(rows: List[PredRow]) -> List[Dict[str, Any]]:
    """Prepare a generic JSONL prediction format that *likely* aligns with
    the official `eval.py` expectations. Adjust keys if needed after you
    inspect `examples/test.jsonl` in your local clone.
    """
    out = []
    for r in rows:
        out.append({
            "language": r.lang,
            "index": r.idx,
            "prediction": r.prediction_raw,
        })
    return out


def aggregate_scores(rows: List[PredRow], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    by_lang: Dict[str, List[PredRow]] = {}
    for r in rows:
        by_lang.setdefault(r.lang, []).append(r)
    summary = {}
    for lang, rr in by_lang.items():
        em = sum(x.em for x in rr) / max(len(rr), 1)
        f1 = sum(x.f1 for x in rr) / max(len(rr), 1)
        summary[lang] = {"EM": em, "F1_char": f1, "n": len(rr)}
    # macro average
    if summary:
        macro_em = sum(v["EM"] for v in summary.values()) / len(summary)
        macro_f1 = sum(v["F1_char"] for v in summary.values()) / len(summary)
    else:
        macro_em = macro_f1 = 0.0
    
    result = {"per_language": summary, "macro": {"EM": macro_em, "F1_char": macro_f1}}
    if system_prompt:
        result["system_prompt"] = system_prompt
    return result


# -------------------------- CLI ----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=False, default=Path("benchmark_data"))
    p.add_argument("--langs", type=str, default="ja", help="Comma-separated language codes, e.g., 'ja,en' ")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Providers
    p.add_argument("--provider", choices=["openai"], default="openai")
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=32)

    # Prompt variants
    p.add_argument("--nospher-prompt-file", type=Path, help="Path to a txt file with your Nospher system prompt (if specified, enables Nospher mode)")

    # Outputs
    p.add_argument("--out-dir", type=Path, default=Path("runs/out"))
    p.add_argument("--tag", type=str, default="dev")

    # Compare
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run folders")
    return p.parse_args()


def build_provider(args: argparse.Namespace) -> LLMProvider:
    if args.provider == "openai":
        return OpenAIProvider(model=args.openai_model, temperature=args.temperature, max_tokens=args.max_tokens)
    raise ValueError("Unknown provider")


def save_run(out_dir: Path, tag: str, rows: List[PredRow], system_prompt: Optional[str] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Raw predictions CSV
    write_csv(out_dir / f"pred_{tag}.csv", rows)
    # JSONL (our schema)
    write_jsonl(out_dir / f"pred_{tag}.jsonl", [dataclasses.asdict(r) for r in rows])
    # Official-style JSONL (adjust keys if needed)
    write_jsonl(out_dir / f"pred_{tag}_official.jsonl", to_official_pred_format(rows))
    # Quick summary
    metrics = aggregate_scores(rows, system_prompt)
    with (out_dir / f"summary_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_compare(dir_a: Path, dir_b: Path) -> None:
    def _load_summary(d: Path) -> Dict[str, Any]:
        cand = list(d.glob("summary_*.json"))
        if not cand:
            raise FileNotFoundError(f"No summary_*.json in {d}")
        with cand[0].open("r", encoding="utf-8") as f:
            return json.load(f)

    A = _load_summary(Path(dir_a))
    B = _load_summary(Path(dir_b))
    def _macro(x):
        return x.get("macro", {}).get("EM", 0.0)
    print("Compare (macro EM):")
    print(f"  A: {_macro(A):.4f}  {dir_a}")
    print(f"  B: {_macro(B):.4f}  {dir_b}")
    print(f"  Diff (B - A): {_macro(B) - _macro(A):.4f}")


async def main_async(args: argparse.Namespace) -> None:
    if args.compare:
        cmd_compare(Path(args.compare[0]), Path(args.compare[1]))
        return

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    provider = build_provider(args)

    # nospher-prompt-fileが指定されていればnospher扱い
    nospher = args.nospher_prompt_file is not None
    nospher_system = None
    
    if nospher:
        if not args.nospher_prompt_file.exists():
            raise FileNotFoundError(f"Nospher prompt file not found: {args.nospher_prompt_file}")
        try:
            nospher_system = args.nospher_prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read nospher prompt file {args.nospher_prompt_file}: {e}") from e
        if not nospher_system.strip():
            raise ValueError(f"Nospher prompt file {args.nospher_prompt_file} is empty")


    rows = await run_eval(
        dataset_root=args.dataset_root,
        langs=langs,
        provider=provider,
        max_samples=args.max_samples if args.max_samples and args.max_samples > 0 else None,
        nospher=nospher,
        nospher_system=nospher_system,
        concurrency=args.concurrency,
        seed=args.seed,
    )

    variant = "nospher" if nospher else "baseline"
    out_dir = args.out_dir
    # 日付時間をフォルダ名に追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / f"{variant}_{'-'.join(langs)}_{args.openai_model.replace('/', '_')}_{timestamp}"
    save_run(out_dir, args.tag, rows)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()

'''
ベースライン実行
# English evaluation
uv run multiloko_nospher_eval.py \
  --dataset-root ./multiloko/benchmark_data \
  --langs english \
  --max-samples 30 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --out-dir runs/baseline

# English evaluation with Nospher
uv run multiloko_nospher_eval.py \
  --dataset-root ./multiloko/benchmark_data \
  --langs english \
  --max-samples 10 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --nospher-prompt-file nospher.txt \
  --out-dir runs/nospher

uv run multiloko_nospher_eval.py \
  --dataset-root ./multiloko/benchmark_data \
  --langs japanese \
  --max-samples 30 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --out-dir runs/baseline

Nospherあり実行
uv run multiloko_nospher_eval.py \
  --dataset-root ./multiloko/benchmark_data \
  --langs japanese \
  --max-samples 30 \
  --provider openai \
  --openai-model gpt-4o-mini \
  --nospher-prompt-file nospher.txt \
  --out-dir runs/nospher
'''