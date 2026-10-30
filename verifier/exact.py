from .checks import k_problem, result, rows_problem, shown, years_problem


def verify_input(value):
    problem = k_problem(value.k) or years_problem(value.first_year, value.last_year)
    if problem:
        return result(False, problem)
    return result(True, "input is valid")


def reference_problem(arguments, reference):
    """What is wrong with the exact_top_terms reference, for the task ``arguments``, or None."""
    ref = reference
    if (ref.dataset, ref.concept, ref.first_year, ref.last_year, ref.k) != (
        arguments.dataset, arguments.concept, arguments.first_year, arguments.last_year, arguments.k
    ):
        return "the reference is for another dataset, concept, year range or k"
    if ref.concept_index < 0:
        return f"the reference's concept_index {ref.concept_index} is negative"
    ids = ref.paper_ids
    if len(set(ids)) != len(ids) or not len(ids) <= ref.scope_papers or ref.scope_papers < 1:
        return "the reference's paper ids are not unique, or more than the collection's N"
    if not ids and ref.ranked_terms:
        return "the reference ranks terms of an empty selection"
    if len(ref.leaders) != min(ref.k + 1, ref.ranked_terms):
        return f"the reference carries {len(ref.leaders)} leaders, not min(k + 1, {ref.ranked_terms})"
    rows = [(term.term, term.tf, term.df) for term in ref.leaders]
    problem = rows_problem(rows, ref.scope_papers)
    if problem:
        return f"the reference's leaders: {problem}"
    if [(t.term, t.tf, t.df, t.score) for t in ref.leaders] != shown(rows, ref.scope_papers):
        return "the reference's displayed scores are not the rule's"
    return None


def verify_output(input_value, output_value, workspace):
    problem = reference_problem(input_value, output_value)
    if problem:
        return result(False, problem)
    return result(True, "the reference is consistent", papers=len(output_value.paper_ids))
