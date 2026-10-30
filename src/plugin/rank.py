"""rank_terms: every term of a selection by TF-IDF, with document frequencies over the chosen scope.

The full ranking is saved as rankings/<paper_set_id>-<idf_scope>.json, each term as [term, tf, df], with the
largest k any call on it returned, so the verifier can tell which terms the model saw.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract import RankedTerm, RankInput, RankOutput
from plugin.bundle import (
    check_k,
    load,
    paper_set_pattern,
    ranking,
    ranking_path,
    select,
    selection_path,
    write_json,
)
from plugin.rule import display


def rank_terms(given: RankInput, workspace: Path) -> RankOutput:
    check_k(given.k)
    if not re.fullmatch(paper_set_pattern(given.dataset), given.paper_set_id):
        raise ValueError(
            f"paper_set_id {given.paper_set_id!r} is not a {given.dataset.value} selection: pass a paper_set_id"
            f" select_papers returned for {given.dataset.value}, <dataset>-c<concept index>-<first_year>-<last_year>"
        )
    path = selection_path(workspace, given.paper_set_id)
    if not path.is_file():
        raise ValueError(f"no selection {given.paper_set_id} is saved: call select_papers for it first")
    saved = json.loads(path.read_text())
    paper_set_id, chosen = select(given.dataset, saved["concept"], saved["first_year"], saved["last_year"])
    papers = load(given.dataset).papers
    if paper_set_id != given.paper_set_id or saved["paper_ids"] != [papers[i].id for i in chosen]:
        raise ValueError(f"the saved selection {given.paper_set_id} was changed: call select_papers for it again")

    n, rows = ranking(given.dataset, chosen, given.idf_scope)
    ranking_id = f"{given.paper_set_id}-{given.idf_scope.value}"
    target = ranking_path(workspace, ranking_id)
    returned = given.k
    if target.is_file():
        returned = max(returned, json.loads(target.read_text())["returned"])
    write_json(
        target,
        {
            "ranking_id": ranking_id,
            "paper_set_id": given.paper_set_id,
            "idf_scope": given.idf_scope.value,
            "selected_papers": len(chosen),
            "scope_papers": n,
            "returned": returned,
            "terms": [[term, tf, df] for term, tf, df, _ in rows],
        },
    )
    return RankOutput(
        ranking_id=ranking_id,
        paper_set_id=given.paper_set_id,
        idf_scope=given.idf_scope,
        selected_papers=len(chosen),
        scope_papers=n,
        ranked_terms=len(rows),
        top=[RankedTerm(term, tf, df, display(value)) for term, tf, df, value in rows[: given.k]],
    )
