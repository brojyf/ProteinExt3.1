from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np


ASPECT_ROOTS = {"P": "GO:0008150", "F": "GO:0003674", "C": "GO:0005575"}


def parse_go_obo(obo_path: Path) -> Dict[str, Set[str]]:
    parents: Dict[str, Set[str]] = {}
    current_id: str | None = None
    current_parents: Set[str] = set()
    is_obsolete = False
    in_term = False
    with Path(obo_path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("["):
                if in_term and current_id and not is_obsolete:
                    parents[current_id] = current_parents
                in_term = line == "[Term]"
                current_id = None
                current_parents = set()
                is_obsolete = False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:]
            elif line.startswith("is_a: "):
                current_parents.add(line[6:].split()[0])
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of" and parts[2].startswith("GO:"):
                    current_parents.add(parts[2])
            elif line == "is_obsolete: true":
                is_obsolete = True
    if in_term and current_id and not is_obsolete:
        parents[current_id] = current_parents
    return parents


def ancestors(term: str, parents: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(parents.get(term, ()))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(parents.get(parent, ()))
    return seen


def propagate_terms(terms: Iterable[str], parents: Dict[str, Set[str]]) -> Set[str]:
    propagated: Set[str] = set()
    for term in terms:
        propagated.add(term)
        propagated.update(ancestors(term, parents))
    return propagated


def build_label_space(
    label_lists: Iterable[Iterable[str]],
    parents: Dict[str, Set[str]],
    *,
    aspect: str,
    min_count: int,
) -> np.ndarray:
    root = ASPECT_ROOTS.get(aspect)
    counts: Counter[str] = Counter()
    for terms in label_lists:
        propagated = propagate_terms(terms, parents)
        if root is not None:
            propagated.discard(root)
        counts.update(propagated)
    return np.asarray(sorted(term for term, count in counts.items() if count >= min_count), dtype=object)


def build_propagation_indices(classes: Sequence[str], parents: Dict[str, Set[str]]) -> List[List[int]]:
    class_to_index = {term: index for index, term in enumerate(classes)}
    descendant_indices: List[List[int]] = [[] for _ in classes]
    for child_index, term in enumerate(classes):
        for parent in ancestors(str(term), parents):
            parent_index = class_to_index.get(parent)
            if parent_index is not None:
                descendant_indices[parent_index].append(child_index)
    return descendant_indices


def propagate_scores(scores: np.ndarray, descendant_indices: List[List[int]]) -> np.ndarray:
    propagated = scores.copy()
    for parent_index, children in enumerate(descendant_indices):
        if children:
            np.maximum(
                propagated[:, parent_index],
                scores[:, children].max(axis=1),
                out=propagated[:, parent_index],
            )
    return propagated
