from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path


DEFAULT_LABELS = Path("training/data/propagated/training.tsv")
DEFAULT_FASTA = Path("training/data/propagated/training.fasta")
DEFAULT_OBO = Path("training/data/go-basic.obo")
DEFAULT_OUTPUT = Path("training/data/ic.pkl")

ASPECTS = ("P", "F", "C")
ASPECT_ROOTS = {
    "P": "GO:0008150",
    "F": "GO:0003674",
    "C": "GO:0005575",
}
NAMESPACE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}
ASPECT_ALIASES = {
    "p": "P",
    "bp": "P",
    "biological_process": "P",
    "f": "F",
    "mf": "F",
    "molecular_function": "F",
    "c": "C",
    "cc": "C",
    "cellular_component": "C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GO information-content pickle for Smin metrics.")
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO, help=f"GO OBO path; default {DEFAULT_OBO}")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA, help=f"training FASTA path; default {DEFAULT_FASTA}")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help=f"training label TSV path; default {DEFAULT_LABELS}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help=f"output pickle path; default {DEFAULT_OUTPUT}")
    return parser.parse_args()


def normalize_protein_id(text: str) -> str:
    token = text.strip().split()[0]
    parts = token.split("|")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return token


def normalize_aspect(text: str) -> str | None:
    return ASPECT_ALIASES.get(text.strip().lower())


def read_fasta_ids(path: Path) -> set[str]:
    protein_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                protein_ids.add(normalize_protein_id(line[1:]))
    return protein_ids


def parse_obo(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    namespaces: dict[str, str] = {}
    current_id: str | None = None
    current_namespace: str | None = None
    is_obsolete = False
    in_term = False

    def flush() -> None:
        if in_term and current_id and not is_obsolete and current_namespace:
            namespaces[current_id] = current_namespace

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("["):
                flush()
                in_term = line == "[Term]"
                current_id = None
                current_namespace = None
                is_obsolete = False
                continue
            if not in_term:
                continue
            if line.startswith("id: GO:"):
                current_id = line[4:]
            elif line.startswith("namespace: "):
                current_namespace = line.removeprefix("namespace: ")
            elif line == "is_obsolete: true":
                is_obsolete = True
            elif line.startswith("is_a: ") and current_id:
                parents[current_id].add(line.removeprefix("is_a: ").split()[0])
            elif line.startswith("relationship: ") and current_id:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of" and parts[2].startswith("GO:"):
                    parents[current_id].add(parts[2])
    flush()
    return dict(parents), namespaces


def ancestors(term: str, parents: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    cached = cache.get(term)
    if cached is not None:
        return cached
    values = {term}
    for parent in parents.get(term, ()):
        values.update(ancestors(parent, parents, cache))
    cache[term] = values
    return values


def read_annotations(path: Path) -> dict[str, dict[str, set[str]]]:
    annotations: dict[str, dict[str, set[str]]] = {
        aspect: defaultdict(set) for aspect in ASPECTS
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"EntryID", "term", "aspect"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"labels TSV must contain columns: {sorted(required)}")
        for row in reader:
            aspect = normalize_aspect(row["aspect"])
            term = row["term"].strip()
            if aspect is not None and term.startswith("GO:"):
                annotations[aspect][normalize_protein_id(row["EntryID"])].add(term)
    return {aspect: dict(values) for aspect, values in annotations.items()}


def build_ic(fasta: Path, labels: Path, obo: Path) -> dict[str, dict[str, float]]:
    protein_ids = read_fasta_ids(fasta)
    if not protein_ids:
        raise ValueError(f"no protein IDs found in FASTA: {fasta}")

    parents, namespaces = parse_obo(obo)
    annotations = read_annotations(labels)
    ancestor_cache: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {aspect: defaultdict(int) for aspect in ASPECTS}

    for protein_id in protein_ids:
        for aspect in ASPECTS:
            propagated = set()
            for term in annotations[aspect].get(protein_id, ()):
                propagated.update(ancestors(term, parents, ancestor_cache))

            for term in propagated:
                if NAMESPACE_TO_ASPECT.get(namespaces.get(term, "")) == aspect:
                    counts[aspect][term] += 1

    protein_count = len(protein_ids)
    ic: dict[str, dict[str, float]] = {}
    for aspect in ASPECTS:
        ic[aspect] = {}
        for term, count in counts[aspect].items():
            ic[aspect][term] = -math.log(count / protein_count)
        ic[aspect][ASPECT_ROOTS[aspect]] = 0.0
    return ic


def main() -> None:
    args = parse_args()
    for path in (args.obo, args.fasta, args.labels):
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")

    ic = build_ic(args.fasta, args.labels, args.obo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (args.fasta.stat().st_mtime, args.labels.stat().st_mtime, ic)
    with args.out.open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"saved: {args.out}")
    for aspect in ASPECTS:
        print(f"{aspect}: {len(ic[aspect])} terms")


if __name__ == "__main__":
    main()
