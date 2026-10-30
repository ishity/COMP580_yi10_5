"""The top-tfidf-terms task's output check, against the exact_top_terms reference and the workspace files.

The first failing check decides: malformed, paper count, provenance, order, then exactness. A malformed
answer, a wrong paper count, a failed provenance check or terms out of the reference's order score 0: a wrong
selection or a wrong scope is the pipeline failing. Only a correct pipeline whose terms are the reference's
top k in order, with some left out, earns partial credit.

The verifier reads exactly two workspace files, named from the task input and the reference's concept index:
selections/<paper_set_id>.json and rankings/<paper_set_id>-collection.json. No other file is listed or read.
"""

from fractions import Fraction
from pathlib import Path

from .checks import RANKINGS, SELECTIONS, read_json, result
from .exact import reference_problem


def paper_set_id(arguments, reference):
    return f"{arguments.dataset.value}-c{reference.concept_index}-{arguments.first_year}-{arguments.last_year}"


def provenance_problem(arguments, reference, terms, workspace):
    """Why the task's selection and its collection-scope ranking are not saved as the reference has them, or None."""
    selection_id = paper_set_id(arguments, reference)
    saved = read_json(Path(workspace) / SELECTIONS / f"{selection_id}.json")
    if saved is None:
        return (
            f"selection {selection_id} (concept {arguments.concept!r}, {arguments.first_year} to"
            f" {arguments.last_year}) is not saved: select_papers was never called for the task's concept and years"
        )
    criteria = (saved["paper_set_id"], saved["dataset"], saved["concept"], saved["first_year"], saved["last_year"])
    task = (selection_id, arguments.dataset.value, arguments.concept, arguments.first_year, arguments.last_year)
    if criteria != task or saved["paper_ids"] != reference.paper_ids:
        return f"the saved selection {selection_id} disagrees with the reference's {len(reference.paper_ids)} papers"
    if not reference.paper_ids:
        if terms:
            return "the selection holds no paper, so no term can be ranked"
        return None
    ranking_id = f"{selection_id}-collection"
    ranked = read_json(Path(workspace) / RANKINGS / f"{ranking_id}.json")
    if ranked is None:
        return (
            f"ranking {ranking_id} is not saved: the selection was never ranked with idf_scope collection, and"
            " document frequencies must come from the whole collection"
        )
    leaders = [[term.term, term.tf, term.df] for term in reference.leaders]
    if (
        (ranked["ranking_id"], ranked["paper_set_id"], ranked["idf_scope"]) != (ranking_id, selection_id, "collection")
        or ranked["scope_papers"] != reference.scope_papers
        or ranked["selected_papers"] != len(reference.paper_ids)
        or len(ranked["terms"]) != reference.ranked_terms
        or ranked["terms"][: len(leaders)] != leaders
    ):
        return f"the saved ranking {ranking_id} disagrees with the reference's counts"
    return None


def subsequence(terms, expected):
    """Whether ``terms`` is ``expected`` with some terms left out, in the same order."""
    remaining = iter(expected)
    return all(term in remaining for term in terms)


def verify_task(arguments, output, reference, workspace):
    """``arguments``: a contract TopTermsInput; ``output``: a TopTermsOutput; ``reference``: an ExactOutput."""
    problem = reference_problem(arguments, reference)
    if problem:
        return result(False, problem)
    terms = list(output.terms)
    papers = len(reference.paper_ids)

    # 1. Malformed.
    if len(set(terms)) != len(terms):
        return result(False, "a term is submitted twice", score=0.0)
    if output.papers < 0:
        return result(False, f"papers cannot be negative, got {output.papers}", score=0.0)
    if len(terms) > arguments.k:
        return result(False, f"{len(terms)} terms are submitted, more than k = {arguments.k}", score=0.0)

    # 2. Paper count.
    if output.papers != papers:
        return result(False, f"the selection holds {papers} papers, not {output.papers}", score=0.0,
                      papers=output.papers, reference_papers=papers)

    # 3. Provenance: the task's selection and its collection-scope ranking, as the reference has them.
    problem = provenance_problem(arguments, reference, terms, workspace)
    if problem:
        return result(False, problem, score=0.0, papers=papers)

    # 4. Order: the terms are the reference's top k, recomputed by the rule, in order with some left out.
    expected = [term.term for term in reference.leaders[: arguments.k]]
    matched = len(set(expected) & set(terms))
    details = {"papers": papers, "submitted": len(terms), "expected": len(expected), "matched": matched}
    if not subsequence(terms, expected):
        outside = [term for term in terms if term not in expected]
        why = f"{outside[0]!r} is not in it" if outside else "they are out of its order"
        return result(False, f"the terms are not the reference's top {len(expected)} in order: {why}", score=0.0,
                      **details)

    # 5. Exactness.
    if terms == expected:
        if not papers:
            return result(True, "no paper of the year range has the concept, so the empty answer is correct", **details)
        return result(True, f"the top {len(expected)} terms of {papers} papers, in order", **details)
    missing = [term for term in expected if term not in terms]
    share = Fraction(matched, len(expected))
    cap = Fraction(len(expected) - 1, len(expected))
    return result(False, f"{len(missing)} of the top {len(expected)} terms are left out, first {missing[0]!r}",
                  score=float(min(share, cap)), **details)
