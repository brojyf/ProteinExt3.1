from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


ASPECT_ALIASES = {
    "p": "p",
    "bp": "p",
    "biological_process": "p",
    "f": "f",
    "mf": "f",
    "molecular_function": "f",
    "c": "c",
    "cc": "c",
    "cellular_component": "c",
}
ASPECTS = ["p", "f", "c"]
THRESHOLDS = np.arange(0, 101, dtype=np.int16) / 100.0


def normalize_aspect(text: str) -> str:
    key = text.strip().lower()
    if key not in ASPECT_ALIASES:
        raise ValueError(f"不支持的 aspect: {text}")
    return ASPECT_ALIASES[key]


def normalize_protein_id(text: str) -> str:
    text = text.strip()
    parts = text.split("|")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return text.split()[0]


def iter_tsv_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split("\t")]
            if len(parts) < 2:
                continue

            first = parts[0].lower()
            if parts[0] in {"AUTHOR", "MODEL", "KEYWORDS", "END"}:
                continue
            if first in {"entryid", "entry", "protein_id", "protein", "query_id", "pid"}:
                continue

            yield parts


def load_labels(path: Path) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    truth_by_aspect: dict[str, dict[str, set[str]]] = {
        aspect: defaultdict(set) for aspect in ASPECTS
    }
    go_to_aspect: dict[str, str] = {}

    for parts in iter_tsv_rows(path):
        if len(parts) < 3:
            raise ValueError(f"label tsv 需要至少三列: pid go_term aspect, 出错文件: {path}")

        protein_id = normalize_protein_id(parts[0])
        go_term = parts[1]
        row_aspect = normalize_aspect(parts[2])
        go_to_aspect[go_term] = row_aspect

        if go_term.startswith("GO:"):
            truth_by_aspect[row_aspect][protein_id].add(go_term)

    return {aspect: dict(proteins) for aspect, proteins in truth_by_aspect.items()}, go_to_aspect


def load_predictions(path: Path, go_to_aspect: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    predictions: dict[str, dict[str, dict[str, float]]] = {
        aspect: defaultdict(dict) for aspect in ASPECTS
    }

    for parts in iter_tsv_rows(path):
        protein_id = normalize_protein_id(parts[0])
        go_term = parts[1]
        if not go_term.startswith("GO:"):
            continue

        row_aspect = None
        score = 1.0

        if len(parts) >= 3:
            token = parts[2].strip().lower()
            if token in ASPECT_ALIASES:
                row_aspect = normalize_aspect(token)
                if len(parts) >= 4:
                    try:
                        score = float(parts[3])
                    except ValueError:
                        score = 1.0
            else:
                try:
                    score = float(parts[2])
                except ValueError:
                    score = 1.0
                if len(parts) >= 4 and parts[3].strip().lower() in ASPECT_ALIASES:
                    row_aspect = normalize_aspect(parts[3])

        if row_aspect is None:
            row_aspect = go_to_aspect.get(go_term)

        if row_aspect not in ASPECTS:
            continue

        old_score = predictions[row_aspect][protein_id].get(go_term)
        if old_score is None or score > old_score:
            predictions[row_aspect][protein_id][go_term] = score

    return {aspect: dict(proteins) for aspect, proteins in predictions.items()}


def per_protein_fmax(truth_terms: set[str], pred_scores: dict[str, float]) -> float:
    if not truth_terms:
        return 0.0

    best = 0.0
    for threshold in THRESHOLDS:
        predicted = {term for term, score in pred_scores.items() if score >= threshold}
        tp = len(truth_terms & predicted)

        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(truth_terms)
        denom = precision + recall
        fscore = (2.0 * precision * recall / denom) if denom > 0 else 0.0
        if fscore > best:
            best = fscore

    return best


def compute_metric_vector(
    proteins: list[str],
    truth_by_protein: dict[str, set[str]],
    predictions: dict[str, dict[str, float]],
) -> np.ndarray:
    values = []
    for protein_id in proteins:
        truth_terms = truth_by_protein.get(protein_id, set())
        pred_scores = predictions.get(protein_id, {})
        values.append(per_protein_fmax(truth_terms, pred_scores))

    return np.asarray(values, dtype=np.float64)


def paired_t_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    if values_a.shape != values_b.shape:
        raise ValueError("两个样本长度不一致，不能做 paired t-test")
    if values_a.size < 2:
        raise ValueError("paired t-test 至少需要 2 个 protein")

    result = ttest_rel(values_a, values_b, alternative="two-sided")
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)

    if math.isnan(t_stat) or math.isnan(p_value):
        diff = values_a - values_b
        mean_diff = float(diff.mean())
        if np.allclose(diff, 0.0):
            return 0.0, 1.0
        return math.copysign(math.inf, mean_diff), 0.0

    return t_stat, p_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取两个 prediction TSV 和一个 label TSV，对 p/f/c 三个 aspect 分别计算 Fmax paired t-test 与 p-value。"
    )
    parser.add_argument("pred_a", help="第一个预测 tsv")
    parser.add_argument("pred_b", help="第二个预测 tsv")
    parser.add_argument("labels", help="label tsv，格式至少包含 pid go_term aspect")
    return parser.parse_args()


def main():
    args = parse_args()

    pred_a_path = Path(args.pred_a)
    pred_b_path = Path(args.pred_b)
    labels_path = Path(args.labels)

    for path in (pred_a_path, pred_b_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"找不到文件: {path}")

    truth_by_aspect, go_to_aspect = load_labels(labels_path)
    pred_a_by_aspect = load_predictions(pred_a_path, go_to_aspect)
    pred_b_by_aspect = load_predictions(pred_b_path, go_to_aspect)

    print("metric\tfmax")
    print("aspect\tn_proteins\tmean_a\tmean_b\tmean_diff\tt_stat\tp_value")

    for aspect in ASPECTS:
        truth_by_protein = truth_by_aspect[aspect]
        if not truth_by_protein:
            print(f"{aspect}\t0\tNA\tNA\tNA\tNA\tNA")
            continue

        proteins = sorted(truth_by_protein)
        values_a = compute_metric_vector(proteins, truth_by_protein, pred_a_by_aspect[aspect])
        values_b = compute_metric_vector(proteins, truth_by_protein, pred_b_by_aspect[aspect])
        t_stat, p_value = paired_t_test(values_a, values_b)

        mean_a = float(values_a.mean())
        mean_b = float(values_b.mean())
        mean_diff = float((values_a - values_b).mean())

        print(
            f"{aspect}\t{len(proteins)}\t{mean_a:.6f}\t{mean_b:.6f}\t"
            f"{mean_diff:.6f}\t{t_stat:.6f}\t{p_value:.6g}"
        )


if __name__ == "__main__":
    main()
