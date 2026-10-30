"""The bundled collections, selections and rankings, and the names and paths the tools share."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from contract import Dataset, IdfScope
from plugin.rule import MAX_K, ranked, terms

DATA = Path(__file__).with_name("data")
SELECTIONS = "selections"
RANKINGS = "rankings"


@dataclass(frozen=True)
class Paper:
    id: str
    year: int
    concepts: frozenset[int]
    # The abstract's terms: term -> how many times it occurs.
    counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Collection:
    concepts: tuple[str, ...]
    papers: tuple[Paper, ...]

    @property
    def first_year(self) -> int:
        return self.papers[0].year

    @property
    def last_year(self) -> int:
        return self.papers[-1].year


@cache
def load(dataset: Dataset) -> Collection:
    raw = json.loads((DATA / f"{dataset.value}.json").read_text())
    concepts = tuple(raw["concepts"])
    if len(set(concepts)) != len(concepts):
        raise ValueError(f"{dataset.value} must hold unique concepts")
    papers = []
    for paper in raw["papers"]:
        held = paper["concepts"]
        if held != sorted(set(held)) or (held and not 0 <= held[0] <= held[-1] < len(concepts)):
            raise ValueError(f"{dataset.value} paper {paper['id']} must list distinct sorted concept indices")
        counts = tuple(sorted(Counter(terms(paper["abstract"])).items()))
        papers.append(Paper(paper["id"], paper["year"], frozenset(held), counts))
    years = [paper.year for paper in papers]
    if not papers or len({paper.id for paper in papers}) != len(papers) or years != sorted(years):
        raise ValueError(f"{dataset.value} must hold papers with unique ids, sorted by year")
    return Collection(concepts=concepts, papers=tuple(papers))


@cache
def document_frequencies(dataset: Dataset) -> Counter:
    """term -> how many papers of the whole collection contain it."""
    return Counter(term for paper in load(dataset).papers for term, _ in paper.counts)


def concept_index(collection: Collection, dataset: Dataset, concept: str) -> int:
    if concept in collection.concepts:
        return collection.concepts.index(concept)
    closest = difflib.get_close_matches(concept, collection.concepts, n=3, cutoff=0)
    raise ValueError(
        f"{concept!r} is not a concept of {dataset.value}: pass a vocabulary concept exactly as written, such as "
        + ", ".join(repr(name) for name in closest)
    )


def check_years(collection: Collection, dataset: Dataset, first_year: int, last_year: int) -> None:
    low, high = collection.first_year, collection.last_year
    if not low <= first_year <= last_year <= high:
        raise ValueError(
            f"years {first_year} to {last_year} are not a range within {dataset.value}'s years {low} to {high}:"
            f" pass first_year and last_year with {low} <= first_year <= last_year <= {high}"
        )


def check_k(k: int) -> None:
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be from 1 to {MAX_K}, got {k}: copy it from the task input")


def select(dataset: Dataset, concept: str, first_year: int, last_year: int) -> tuple[str, list[int]]:
    """The selection's paper_set_id and its paper indices, in dataset order."""
    collection = load(dataset)
    index = concept_index(collection, dataset, concept)
    check_years(collection, dataset, first_year, last_year)
    chosen = [
        i for i, paper in enumerate(collection.papers)
        if index in paper.concepts and first_year <= paper.year <= last_year
    ]
    return f"{dataset.value}-c{index}-{first_year}-{last_year}", chosen


def ranking(dataset: Dataset, chosen: list[int], scope: IdfScope) -> tuple[int, list[tuple]]:
    """N and every term of the chosen papers as (term, tf, df, score), in ranking order."""
    papers = load(dataset).papers
    tf: Counter = Counter()
    for i in chosen:
        tf.update(dict(papers[i].counts))
    if scope is IdfScope.COLLECTION:
        n, df = len(papers), document_frequencies(dataset)
    else:
        n, df = len(chosen), Counter(term for i in chosen for term, _ in papers[i].counts)
    return n, ranked({term: (count, df[term]) for term, count in tf.items()}, n)


def paper_set_pattern(dataset: Dataset) -> str:
    return re.escape(dataset.value) + r"-c(0|[1-9][0-9]*)-([0-9]{4})-([0-9]{4})"


def selection_path(workspace: Path, paper_set_id: str) -> Path:
    return workspace / SELECTIONS / f"{paper_set_id}.json"


def ranking_path(workspace: Path, ranking_id: str) -> Path:
    return workspace / RANKINGS / f"{ranking_id}.json"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")
