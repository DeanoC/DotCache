from __future__ import annotations

import json
import hashlib
import re
import random
import string
import zipfile
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - exercised via fallback path in tests.
    jieba = None


OFFICIAL_PROMPTS: dict[str, str] = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

OFFICIAL_MAX_NEW_TOKENS: dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

OFFICIAL_METRICS: dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

TASK_FAMILIES: dict[str, str] = {
    "narrativeqa": "qa",
    "qasper": "qa",
    "multifieldqa_en": "qa",
    "multifieldqa_zh": "qa",
    "hotpotqa": "qa",
    "2wikimqa": "qa",
    "musique": "qa",
    "dureader": "qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "vcsum": "summarization",
    "trec": "classification",
    "triviaqa": "qa",
    "samsum": "summarization",
    "lsht": "classification",
    "passage_count": "counting",
    "passage_retrieval_en": "retrieval",
    "passage_retrieval_zh": "retrieval",
    "lcc": "code",
    "repobench-p": "code",
}

FIRST_LINE_DATASETS = frozenset({"trec", "triviaqa", "samsum", "lsht"})


@dataclass(frozen=True, slots=True)
class LongBenchDatasetSpec:
    dataset: str
    prompt_template: str
    max_new_tokens: int
    metric_name: str
    task_family: str


DATASET_SPECS: dict[str, LongBenchDatasetSpec] = {
    dataset: LongBenchDatasetSpec(
        dataset=dataset,
        prompt_template=OFFICIAL_PROMPTS[dataset],
        max_new_tokens=int(OFFICIAL_MAX_NEW_TOKENS[dataset]),
        metric_name=OFFICIAL_METRICS[dataset],
        task_family=TASK_FAMILIES[dataset],
    )
    for dataset in OFFICIAL_PROMPTS
}


def list_supported_datasets() -> tuple[str, ...]:
    return tuple(DATASET_SPECS)


def get_dataset_spec(dataset: str) -> LongBenchDatasetSpec:
    if dataset not in DATASET_SPECS:
        raise KeyError(f"unsupported LongBench dataset: {dataset}")
    return DATASET_SPECS[dataset]


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text: str) -> str:
    def white_space_fix(value: str) -> str:
        return "".join(value.split())

    def remove_punc(value: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in value if ch not in all_punctuation)

    return white_space_fix(remove_punc(text.lower()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        curr = [0]
        for right_index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                curr.append(prev[right_index - 1] + 1)
            else:
                curr.append(max(prev[right_index], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    lcs = _lcs_length(prediction_tokens, ground_truth_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return _f1_score(prediction_tokens, ground_truth_tokens)


def _zh_tokens(text: str) -> list[str]:
    if jieba is not None:
        tokens = [normalize_zh_answer(token) for token in jieba.cut(text, cut_all=False)]
        return [token for token in tokens if token]
    normalized = normalize_zh_answer(text)
    return [char for char in normalized if char]


def qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    return _f1_score(_zh_tokens(prediction), _zh_tokens(ground_truth))


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    return _rouge_l_f1(prediction.split(), ground_truth.split())


def rouge_l_zh_score(prediction: str, ground_truth: str) -> float:
    return _rouge_l_f1(_zh_tokens(prediction), _zh_tokens(ground_truth))


def classification_score(prediction: str, ground_truth: str, *, all_classes: list[str]) -> float:
    matches = [class_name for class_name in all_classes if class_name in prediction]
    filtered: list[str] = []
    for match in matches:
        if match in ground_truth and match != ground_truth:
            continue
        filtered.append(match)
    if ground_truth not in filtered:
        return 0.0
    return 1.0 / len(filtered)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if number == str(ground_truth_id))
    return right / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if number == str(ground_truth_id))
    return right / len(numbers)


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if number == str(ground_truth))
    return right / len(numbers)


def code_sim_score(prediction: str, ground_truth: str) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").splitlines():
        stripped = line.strip()
        if "`" in stripped or "#" in stripped or "//" in stripped:
            continue
        candidate = stripped
        break
    return SequenceMatcher(None, candidate, ground_truth).ratio()


def _metric_fn(metric_name: str):
    mapping = {
        "qa_f1": lambda pred, gt, all_classes: qa_f1_score(pred, gt),
        "qa_f1_zh": lambda pred, gt, all_classes: qa_f1_zh_score(pred, gt),
        "rouge_l": lambda pred, gt, all_classes: rouge_l_score(pred, gt),
        "rouge_l_zh": lambda pred, gt, all_classes: rouge_l_zh_score(pred, gt),
        "classification": lambda pred, gt, all_classes: classification_score(
            pred,
            gt,
            all_classes=all_classes,
        ),
        "retrieval": lambda pred, gt, all_classes: retrieval_score(pred, gt),
        "retrieval_zh": lambda pred, gt, all_classes: retrieval_zh_score(pred, gt),
        "count": lambda pred, gt, all_classes: count_score(pred, gt),
        "code_sim": lambda pred, gt, all_classes: code_sim_score(pred, gt),
    }
    return mapping[metric_name]


def _prepare_prediction_for_metric(dataset: str, prediction: str) -> str:
    candidate = str(prediction)
    if dataset in FIRST_LINE_DATASETS:
        candidate = candidate.lstrip("\n").split("\n")[0]
    return candidate.strip()


def score_prediction(
    dataset: str,
    prediction: str,
    answers: list[str],
    *,
    all_classes: list[str] | None = None,
) -> dict[str, Any]:
    spec = get_dataset_spec(dataset)
    candidate = _prepare_prediction_for_metric(dataset, prediction)
    metric_fn = _metric_fn(spec.metric_name)

    best_score = 0.0
    best_answer = ""
    for answer in answers:
        score = float(metric_fn(candidate, str(answer), list(all_classes or [])))
        if score > best_score:
            best_score = score
            best_answer = str(answer)

    return {
        "longbench_prediction_scored": candidate,
        "longbench_metric_name": spec.metric_name,
        "longbench_task_family": spec.task_family,
        "longbench_official_score": float(best_score),
        "longbench_best_matching_answer_official": best_answer,
    }


def count_dataset_rows(zip_path: Path, dataset: str) -> int:
    member_name = f"data/{dataset}.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            return sum(1 for _ in handle)


def load_dataset_rows(zip_path: Path, dataset: str) -> list[dict[str, Any]]:
    member_name = f"data/{dataset}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            for raw_line in handle:
                rows.append(json.loads(raw_line))
    return rows


def stratified_row_indices(row_count: int, sample_count: int) -> list[int]:
    total_rows = int(row_count)
    requested = int(sample_count)
    if total_rows <= 0 or requested <= 0:
        return []
    if requested >= total_rows:
        return list(range(total_rows))
    if requested == 1:
        return [0]

    indices: list[int] = []
    seen: set[int] = set()
    for sample_index in range(requested):
        position = round(sample_index * (total_rows - 1) / float(requested - 1))
        row_index = int(position)
        if row_index not in seen:
            indices.append(row_index)
            seen.add(row_index)

    if len(indices) < requested:
        for row_index in range(total_rows):
            if row_index in seen:
                continue
            indices.append(row_index)
            seen.add(row_index)
            if len(indices) >= requested:
                break
    return sorted(indices[:requested])


def _stable_random(seed: int, dataset: str, quartile_index: int) -> random.Random:
    seed_material = f"{int(seed)}:{dataset}:{int(quartile_index)}".encode("utf-8")
    seed_value = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    return random.Random(seed_value)


def build_length_quartile_prompt_specs_from_zip(
    zip_path: Path,
    *,
    datasets: list[str] | tuple[str, ...] | None = None,
    rows_per_quartile: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    selected = list(datasets or list_supported_datasets())
    prompt_specs: list[dict[str, Any]] = []
    target_per_dataset = max(int(rows_per_quartile), 0) * 4
    for dataset in selected:
        spec = get_dataset_spec(dataset)
        rows = load_dataset_rows(zip_path, dataset)
        ranked_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            ranked_rows.append(
                {
                    "row_index": int(row_index),
                    "row_length": int(row.get("length", 0) or 0),
                }
            )
        ranked_rows.sort(key=lambda item: (int(item["row_length"]), int(item["row_index"])))
        selected_rows: list[dict[str, Any]] = []
        selected_indices: set[int] = set()
        for quartile_index in range(4):
            start = (quartile_index * len(ranked_rows)) // 4
            end = ((quartile_index + 1) * len(ranked_rows)) // 4 if quartile_index < 3 else len(ranked_rows)
            bucket = ranked_rows[start:end]
            if not bucket:
                continue
            sample_size = min(int(rows_per_quartile), len(bucket))
            rng = _stable_random(seed, dataset, quartile_index)
            sample_positions = sorted(rng.sample(range(len(bucket)), k=sample_size))
            for position in sample_positions:
                row = dict(bucket[position])
                row["length_quartile"] = quartile_index
                selected_rows.append(row)
                selected_indices.add(int(row["row_index"]))
        if len(selected_rows) < target_per_dataset:
            remaining = [row for row in ranked_rows if int(row["row_index"]) not in selected_indices]
            for row in remaining[: max(target_per_dataset - len(selected_rows), 0)]:
                selected_rows.append(
                    {
                        "row_index": int(row["row_index"]),
                        "row_length": int(row["row_length"]),
                        "length_quartile": None,
                    }
                )
        selected_rows.sort(key=lambda item: int(item["row_index"]))
        for row in selected_rows[:target_per_dataset]:
            prompt_specs.append(
                {
                    "prompt_id": f"{dataset}_row{int(row['row_index'])}",
                    "dataset": dataset,
                    "row_index": int(row["row_index"]),
                    "task_family": spec.task_family,
                    "metric_name": spec.metric_name,
                    "row_length": int(row["row_length"]),
                    "length_quartile": row["length_quartile"],
                }
            )
    return prompt_specs


def build_prompt_specs_from_zip(
    zip_path: Path,
    *,
    datasets: list[str] | tuple[str, ...] | None = None,
    limit_per_dataset: int | None = None,
    stratified_limit_per_dataset: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(datasets or list_supported_datasets())
    prompt_specs: list[dict[str, Any]] = []
    for dataset in selected:
        spec = get_dataset_spec(dataset)
        row_count = count_dataset_rows(zip_path, dataset)
        if stratified_limit_per_dataset is not None:
            row_indices = stratified_row_indices(row_count, min(row_count, int(stratified_limit_per_dataset)))
        else:
            limit = row_count if limit_per_dataset is None else min(row_count, int(limit_per_dataset))
            row_indices = list(range(limit))
        for row_index in row_indices:
            prompt_specs.append(
                {
                    "prompt_id": f"{dataset}_row{row_index}",
                    "dataset": dataset,
                    "row_index": row_index,
                    "task_family": spec.task_family,
                    "metric_name": spec.metric_name,
                }
            )
    return prompt_specs
