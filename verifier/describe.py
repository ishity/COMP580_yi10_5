from .checks import result


def verify_input(value):
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    out = output_value
    if out.dataset != input_value.dataset:
        return result(False, f"output describes {out.dataset.value}, not {input_value.dataset.value}")
    if out.papers < 1 or out.first_year > out.last_year:
        return result(False, "a collection needs a paper, and its first year cannot follow its last")
    if not 0 <= out.concepts_with_papers <= out.concepts or out.terms < 0:
        return result(False, "the concepts with papers are some of the concepts, and terms cannot be negative")
    return result(True, "the collection is described", papers=out.papers, terms=out.terms)
