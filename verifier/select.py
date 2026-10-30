from pathlib import Path

from .checks import SELECTIONS, read_json, result, years_problem


def verify_input(value):
    problem = years_problem(value.first_year, value.last_year)
    if problem:
        return result(False, problem)
    if not value.concept.strip():
        return result(False, "concept must name a concept of the dataset")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    out = output_value
    given = (input_value.dataset, input_value.concept, input_value.first_year, input_value.last_year)
    if (out.dataset, out.concept, out.first_year, out.last_year) != given:
        return result(False, "the output echoes another dataset, concept or year range than the input")
    prefix = f"{out.dataset.value}-c"
    suffix = f"-{out.first_year}-{out.last_year}"
    middle = out.paper_set_id[len(prefix) : len(out.paper_set_id) - len(suffix)]
    if not (out.paper_set_id.startswith(prefix) and out.paper_set_id.endswith(suffix) and middle.isdigit()):
        return result(False, f"paper_set_id {out.paper_set_id!r} is not {prefix}<index>{suffix}")
    saved = read_json(Path(workspace) / SELECTIONS / f"{out.paper_set_id}.json")
    if saved is None:
        return result(False, f"selection {out.paper_set_id} is not saved in the workspace")
    if (saved["paper_set_id"], saved["dataset"], saved["concept"], saved["first_year"], saved["last_year"]) != (
        out.paper_set_id, out.dataset.value, out.concept, out.first_year, out.last_year
    ):
        return result(False, "the saved selection is for another id, dataset, concept or year range")
    ids = saved["paper_ids"]
    if len(set(ids)) != len(ids) or len(ids) != out.papers:
        return result(False, f"the saved selection holds {len(ids)} papers (unique: {len(set(ids))}), not {out.papers}")
    return result(True, "the selection is saved", papers=out.papers)
