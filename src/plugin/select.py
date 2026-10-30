"""select_papers: the papers of a year range whose concept set holds a concept, saved under a paper_set_id."""

from __future__ import annotations

from pathlib import Path

from contract import SelectInput, SelectOutput
from plugin.bundle import load, select, selection_path, write_json


def select_papers(given: SelectInput, workspace: Path) -> SelectOutput:
    paper_set_id, chosen = select(given.dataset, given.concept, given.first_year, given.last_year)
    papers = load(given.dataset).papers
    write_json(
        selection_path(workspace, paper_set_id),
        {
            "paper_set_id": paper_set_id,
            "dataset": given.dataset.value,
            "concept": given.concept,
            "first_year": given.first_year,
            "last_year": given.last_year,
            "paper_ids": [papers[i].id for i in chosen],
        },
    )
    return SelectOutput(
        paper_set_id=paper_set_id,
        dataset=given.dataset,
        concept=given.concept,
        first_year=given.first_year,
        last_year=given.last_year,
        papers=len(chosen),
    )
