from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np


def parse_go_obo(obo_path: Path) -> Dict[str, Set[str]]:
    """
    Parse a GO OBO file and return direct parent relationships.
    Includes is_a and part_of relationships (standard CAFA propagation rules).

    Returns
    -------
    parents : dict[term_id, set[parent_term_id]]
        Direct parent IDs for each non-obsolete GO term.
    """
    parents: Dict[str, Set[str]] = {}
    current_id: str | None = None
    current_parents: Set[str] = set()
    is_obsolete = False
    in_term = False

    with open(obo_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line.startswith("["):
                if in_term and current_id and not is_obsolete:
                    parents[current_id] = current_parents
                in_term = line == "[Term]"
                current_id = None
                current_parents = set()
                is_obsolete = False

            elif not in_term:
                continue

            elif line.startswith("id: "):
                current_id = line[4:]

            elif line.startswith("is_a: "):
                parent_id = line[6:].split()[0]
                current_parents.add(parent_id)

            elif line.startswith("relationship:"):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of" and parts[2].startswith("GO:"):
                    current_parents.add(parts[2])

            elif line == "is_obsolete: true":
                is_obsolete = True

    if in_term and current_id and not is_obsolete:
        parents[current_id] = current_parents

    return parents


def _all_ancestors(term: str, parents: Dict[str, Set[str]]) -> Set[str]:
    """Return the full transitive ancestor set of *term* (DFS, no recursion limit)."""
    ancestors: Set[str] = set()
    stack = list(parents.get(term, ()))
    while stack:
        node = stack.pop()
        if node not in ancestors:
            ancestors.add(node)
            stack.extend(parents.get(node, ()))
    return ancestors


def build_propagation_indices(
    classes: np.ndarray,
    parents: Dict[str, Set[str]],
) -> List[List[int]]:
    """
    Precompute, for each class index *j*, the list of descendant class indices.

    ``descendant_indices[j]`` contains the column indices of all terms whose
    annotations/scores should propagate *up* to ``classes[j]``.  That is,
    every term that has ``classes[j]`` as an ancestor.

    Calling ``propagate_scores`` with the returned list then runs in O(n_classes)
    Python iterations, each doing a single vectorised numpy max over a batch of
    descendant columns — much faster than per-pair iteration.
    """
    class_list: List[str] = classes.tolist()
    class_to_idx: Dict[str, int] = {term: idx for idx, term in enumerate(class_list)}
    n_classes = len(class_list)
    descendant_indices: List[List[int]] = [[] for _ in range(n_classes)]

    for child_idx, term in enumerate(class_list):
        for anc in _all_ancestors(term, parents):
            anc_idx = class_to_idx.get(anc)
            if anc_idx is not None:
                descendant_indices[anc_idx].append(child_idx)

    return descendant_indices


def propagate_scores(
    scores: np.ndarray,
    descendant_indices: List[List[int]],
) -> np.ndarray:
    """
    Propagate scores upward through the GO DAG.

    For each term, the propagated score equals:
        max(own score, max score of all its descendants)

    This reflects the CAFA evaluation convention: predicting a specific term
    implicitly predicts all its ancestors.

    Parameters
    ----------
    scores : (n_samples, n_classes) float array
    descendant_indices : precomputed from :func:`build_propagation_indices`

    Returns
    -------
    propagated : (n_samples, n_classes) float array (copy of *scores*)
    """
    propagated = scores.copy()
    for anc_idx, desc_idxs in enumerate(descendant_indices):
        if desc_idxs:
            np.maximum(
                propagated[:, anc_idx],
                scores[:, desc_idxs].max(axis=1),
                out=propagated[:, anc_idx],
            )
    return propagated
