"""describe_collection: how many papers, years, distinct terms and concepts a bundled collection holds."""

from __future__ import annotations

from pathlib import Path

from contract import DescribeInput, DescribeOutput
from plugin.bundle import document_frequencies, load


def describe_collection(given: DescribeInput, workspace: Path) -> DescribeOutput:
    collection = load(given.dataset)
    return DescribeOutput(
        dataset=given.dataset,
        papers=len(collection.papers),
        first_year=collection.first_year,
        last_year=collection.last_year,
        terms=len(document_frequencies(given.dataset)),
        concepts=len(collection.concepts),
        concepts_with_papers=len(set().union(*(paper.concepts for paper in collection.papers))),
    )
