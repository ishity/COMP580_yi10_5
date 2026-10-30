"""exact_top_terms: the task's reference, the collection-scope ranking of the task's selection, out of the
model's sight. It carries the concept index (so the verifier can name the selection and ranking files), the selected
paper ids and the first k + 1 terms with their counts."""

from __future__ import annotations

from pathlib import Path

from contract import ExactOutput, IdfScope, RankedTerm, TopTermsInput
from plugin.bundle import check_k, concept_index, load, ranking, select
from plugin.rule import display


def exact_top_terms(given: TopTermsInput, workspace: Path) -> ExactOutput:
    check_k(given.k)
    _, chosen = select(given.dataset, given.concept, given.first_year, given.last_year)
    n, rows = ranking(given.dataset, chosen, IdfScope.COLLECTION)
    papers = load(given.dataset).papers
    return ExactOutput(
        dataset=given.dataset,
        concept=given.concept,
        concept_index=concept_index(load(given.dataset), given.dataset, given.concept),
        first_year=given.first_year,
        last_year=given.last_year,
        k=given.k,
        paper_ids=[papers[i].id for i in chosen],
        scope_papers=n,
        ranked_terms=len(rows),
        leaders=[RankedTerm(term, tf, df, display(value)) for term, tf, df, value in rows[: given.k + 1]],
    )
