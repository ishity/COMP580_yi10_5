from pathlib import Path

from .checks import (
    RANKINGS,
    SELECTIONS,
    k_problem,
    paper_set_problem,
    read_json,
    result,
    rows_problem,
    shown,
)


def verify_input(value):
    problem = k_problem(value.k) or paper_set_problem(value.dataset, value.paper_set_id)
    if problem:
        return result(False, problem)
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    out = output_value
    if (out.paper_set_id, out.idf_scope) != (input_value.paper_set_id, input_value.idf_scope):
        return result(False, "the output is for another paper_set_id or idf_scope than the input")
    if out.ranking_id != f"{out.paper_set_id}-{out.idf_scope.value}":
        return result(False, f"ranking_id {out.ranking_id!r} is not {out.paper_set_id}-{out.idf_scope.value}")
    selection = read_json(Path(workspace) / SELECTIONS / f"{out.paper_set_id}.json")
    if selection is None:
        return result(False, f"selection {out.paper_set_id} is not saved in the workspace")
    saved = read_json(Path(workspace) / RANKINGS / f"{out.ranking_id}.json")
    if saved is None:
        return result(False, f"ranking {out.ranking_id} is not saved in the workspace")
    if (saved["ranking_id"], saved["paper_set_id"], saved["idf_scope"]) != (
        out.ranking_id, out.paper_set_id, out.idf_scope.value
    ):
        return result(False, "the saved ranking is for another id, selection or scope")
    selected = len(selection["paper_ids"])
    n = saved["scope_papers"]
    if (out.selected_papers, saved["selected_papers"], out.scope_papers) != (selected, selected, n):
        return result(False, f"the selection holds {selected} papers, and the saved ranking's N is {n}")
    in_selection = out.idf_scope.value == "selection"
    if (in_selection and n != selected) or (not in_selection and (n < selected or n < 1)):
        return result(False, f"a {out.idf_scope.value} scope of {selected} selected papers cannot hold N = {n}")
    rows = [tuple(row) for row in saved["terms"]]
    if out.ranked_terms != len(rows) or (not selected and rows):
        return result(False, f"the saved ranking holds {len(rows)} terms, not {out.ranked_terms}")
    problem = rows_problem(rows, n, selected if in_selection else None)
    if problem:
        return result(False, problem)
    if saved["returned"] < input_value.k:
        return result(False, f"the saved ranking records {saved['returned']} returned terms, fewer than k")
    top = [(term.term, term.tf, term.df, term.score) for term in out.top]
    if top != shown(rows[: input_value.k], n):
        return result(False, "the top terms are not the saved ranking's first k, with their scores")
    return result(True, "the ranking is rederived", terms=len(rows), scope_papers=n)
