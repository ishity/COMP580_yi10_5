from __future__ import annotations

import json
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import TASKS, TOOLS, Dataset, ExactOutput, IdfScope, RankInput, SelectInput, Task, Tool, TopTermsInput
from plugin import exact_top_terms, rank_terms, select_papers

A = {"dataset": "acl-small", "concept": "question answering", "first_year": 2018, "last_year": 2019, "k": 10}
B = {"dataset": "acl-small", "concept": "named entity recognition", "first_year": 2019, "last_year": 2019, "k": 15}
C = {"dataset": "acl-small", "concept": "masked language modeling", "first_year": 2018, "last_year": 2018, "k": 5}
D = {"dataset": "edge-cases", "concept": "parsing", "first_year": 2020, "last_year": 2021, "k": 5}
D_TOP = ["parser", "tree", "parsing", "chart", "grammar"]
# Each main instance with its year range off by one.
OFF_BY_ONE = {"A": A | {"first_year": 2019}, "B": B | {"first_year": 2018}, "C": C | {"last_year": 2019},
              "D": D | {"first_year": 2021}}


def reference(arguments: dict) -> dict:
    return plain_output(exact_top_terms(build_output(arguments, TopTermsInput), Path("unused")))


def verify_task(arguments: dict, terms: list[str], papers: int, workspace: Path, ref: dict | None = None) -> dict:
    """The verdict on (terms, papers) for the task ``arguments``, against its exact_top_terms reference."""
    given, made = TASKS[Task.TOP_TFIDF_TERMS]
    return verifier.verify_task(
        Task.TOP_TFIDF_TERMS.value,
        build_output(arguments, given),
        build_output({"terms": terms, "papers": papers}, made),
        build_output(ref or reference(arguments), ExactOutput),
        str(workspace),
    )


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def pipeline(workspace: Path, run: dict, scope: IdfScope = IdfScope.COLLECTION, k: int | None = None):
    """Select ``run``'s papers and rank them over ``scope`` with ``k`` (run's own k when None); the submission."""
    dataset = Dataset(run["dataset"])
    chosen = select_papers(SelectInput(dataset, run["concept"], run["first_year"], run["last_year"]), workspace)
    if not chosen.papers:
        return [], 0
    made = rank_terms(RankInput(dataset, chosen.paper_set_id, scope, k or run["k"]), workspace)
    return [term.term for term in made.top][: run["k"]], chosen.papers


def edit(path: Path, change) -> None:
    saved = json.loads(path.read_text())
    change(saved)
    path.write_text(json.dumps(saved))


@pytest.mark.parametrize("arguments", [A, B, C, D], ids=["A", "B", "C", "D"])
def test_the_correct_pipeline_passes(tmp_path: Path, arguments: dict) -> None:
    terms, papers = pipeline(tmp_path, arguments)
    verdict = verify_task(arguments, terms, papers, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0
    assert verdict == verify_task(arguments, terms, papers, tmp_path)  # the same verdict twice


def test_instance_d_breaks_its_three_way_tie_by_term(tmp_path: Path) -> None:
    terms, papers = pipeline(tmp_path, D)
    assert (terms, papers) == (D_TOP, 3)
    assert [term["term"] for term in reference(D)["leaders"]] == [*D_TOP, "lattice"]


@pytest.mark.parametrize("arguments", [A, B, D], ids=["A", "B", "D"])
def test_the_selection_scope_scores_zero(tmp_path: Path, arguments: dict) -> None:
    terms, papers = pipeline(tmp_path, arguments, IdfScope.SELECTION)
    assert terms != [term["term"] for term in reference(arguments)["leaders"][: arguments["k"]]]
    verdict = verify_task(arguments, terms, papers, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0
    assert "never ranked with idf_scope collection" in verdict["message"]


def test_the_scope_cannot_matter_for_an_empty_selection(tmp_path: Path) -> None:
    select_papers(SelectInput(Dataset.ACL_SMALL, C["concept"], 2018, 2018), tmp_path)
    assert verify_task(C, [], 0, tmp_path)["valid"] is True


@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_a_year_range_off_by_one_scores_zero(tmp_path: Path, name: str) -> None:
    arguments = {"A": A, "B": B, "C": C, "D": D}[name]
    terms, papers = pipeline(tmp_path, OFF_BY_ONE[name])
    verdict = verify_task(arguments, terms, papers, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "papers, not" in verdict["message"]


def test_the_right_count_from_an_off_by_one_selection_fails_provenance(tmp_path: Path) -> None:
    terms, _ = pipeline(tmp_path, OFF_BY_ONE["D"])
    verdict = verify_task(D, terms, 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "is not saved" in verdict["message"]


def test_an_empty_answer_needs_the_empty_selection(tmp_path: Path) -> None:
    verdict = verify_task(C, [], 0, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "select_papers was never called" in verdict["message"]
    pipeline(tmp_path, C)
    assert not (tmp_path / "rankings").exists()  # no rank_terms call is needed
    verdict = verify_task(C, [], 0, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0


def test_terms_for_an_empty_selection_fail(tmp_path: Path) -> None:
    pipeline(tmp_path, C)
    verdict = verify_task(C, ["masked"], 0, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "no paper" in verdict["message"]


def test_an_empty_answer_fails_when_papers_match(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    verdict = verify_task(D, [], 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0


@pytest.mark.parametrize(
    ("terms", "papers", "message"),
    [([*D_TOP[:4], "parser"], 3, "submitted twice"), (D_TOP, -1, "cannot be negative"),
     ([*D_TOP, "lattice"], 3, "more than k = 5")],
    ids=["duplicate", "negative", "too-many"],
)
def test_refuses_a_malformed_answer(tmp_path: Path, terms: list[str], papers: int, message: str) -> None:
    pipeline(tmp_path, D, k=6)
    verdict = verify_task(D, terms, papers, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and message in verdict["message"]


def test_without_a_ranking_the_answer_scores_zero(tmp_path: Path) -> None:
    select_papers(SelectInput(Dataset.EDGE_CASES, "parsing", 2020, 2021), tmp_path)
    verdict = verify_task(D, D_TOP, 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "never ranked" in verdict["message"]


def test_exploratory_files_are_never_read(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    pipeline(tmp_path, OFF_BY_ONE["D"])  # another selection and ranking, beside the task's
    (tmp_path / "selections" / "notes.json").write_text("not json")
    (tmp_path / "rankings" / "edge-cases-c1-2020-2021-collection.json").write_text("not json")
    assert verify_task(D, D_TOP, 3, tmp_path)["valid"] is True


def test_a_wrong_concept_index_in_the_reference_finds_no_selection(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    ref = reference(D)
    ref["concept_index"] = 1
    verdict = verify_task(D, D_TOP, 3, tmp_path, ref)
    assert verdict["valid"] is False and "edge-cases-c1-2020-2021" in verdict["message"]


def test_a_term_outside_the_top_k_scores_zero(tmp_path: Path) -> None:
    pipeline(tmp_path, D, k=30)  # lattice, rank 6, was returned, but is not in the top 5
    verdict = verify_task(D, [*D_TOP[:4], "lattice"], 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "'lattice' is not in it" in verdict["message"]


def test_a_larger_k_call_backs_the_top_k(tmp_path: Path) -> None:
    pipeline(tmp_path, D, k=30)
    assert verify_task(D, D_TOP, 3, tmp_path)["valid"] is True


def test_a_correct_pipeline_with_a_shorter_list_earns_its_share(tmp_path: Path) -> None:
    pipeline(tmp_path, A)
    top = [term["term"] for term in reference(A)["leaders"][:10]]
    verdict = verify_task(A, top[:7], 288, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == pytest.approx(0.7)
    assert verdict["details"] == {"papers": 288, "submitted": 7, "expected": 10, "matched": 7}


def test_one_dropped_term_earns_the_capped_share(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    verdict = verify_task(D, [term for term in D_TOP if term != "parsing"], 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == pytest.approx(0.8)
    assert "1 of the top 5 terms are left out, first 'parsing'" in verdict["message"]
    terms, papers = pipeline(tmp_path, A)
    verdict = verify_task(A, terms[:4] + terms[5:], papers, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == pytest.approx(0.9)  # 9 of 10, at the cap


def test_two_swapped_terms_score_zero(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    verdict = verify_task(D, ["tree", "parser", *D_TOP[2:]], 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "out of its order" in verdict["message"]
    verdict = verify_task(D, ["parser", "tree", "parsing", "grammar", "chart"], 3, tmp_path)  # tied, swapped
    assert verdict["valid"] is False and verdict["score"] == 0.0


def test_a_tie_broken_the_wrong_way_scores_zero(tmp_path: Path) -> None:
    pipeline(tmp_path, D, k=6)
    verdict = verify_task(D, [*D_TOP[:4], "lattice"], 3, tmp_path)  # lattice ties grammar, and sorts after it
    assert verdict["valid"] is False and verdict["score"] == 0.0


def test_a_selection_scope_list_after_a_large_k_collection_ranking_scores_zero(tmp_path: Path) -> None:
    for arguments in (A, B, D):
        workspace = tmp_path / arguments["concept"]
        pipeline(workspace, arguments, k=30)  # every selection-scope top term is among the 30 returned
        wrong, papers = pipeline(workspace, arguments, IdfScope.SELECTION)
        verdict = verify_task(arguments, wrong, papers, workspace)
        assert verdict["valid"] is False and verdict["score"] == 0.0 and "top" in verdict["message"]


@pytest.mark.parametrize(
    "change",
    [lambda saved: saved["terms"][0].__setitem__(1, 5), lambda saved: saved.update(scope_papers=9),
     lambda saved: saved["terms"].pop()],
    ids=["tf", "n", "count"],
)
def test_a_tampered_ranking_fails_provenance(tmp_path: Path, change) -> None:
    pipeline(tmp_path, D)
    edit(tmp_path / "rankings" / "edge-cases-c0-2020-2021-collection.json", change)
    verdict = verify_task(D, D_TOP, 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "disagrees with the reference" in verdict["message"]


def test_a_selection_with_other_papers_fails_provenance(tmp_path: Path) -> None:
    pipeline(tmp_path, D)
    edit(tmp_path / "selections" / "edge-cases-c0-2020-2021.json", lambda saved: saved["paper_ids"].pop())
    verdict = verify_task(D, D_TOP, 3, tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "disagrees with the reference" in verdict["message"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ref: ref["leaders"].reverse(), "not ranked by score"),
        (lambda ref: ref["leaders"].pop(), "leaders, not min(k + 1"),
        (lambda ref: ref["leaders"][0].update(score="9.000000"), "displayed scores"),
        (lambda ref: ref.update(k=4), "another dataset, concept"),
        (lambda ref: ref.update(concept_index=-1), "concept_index -1 is negative"),
    ],
    ids=["order", "short", "score", "k", "index"],
)
def test_refuses_an_inconsistent_reference(tmp_path: Path, change, message: str) -> None:
    terms, papers = pipeline(tmp_path, D)
    ref = reference(D)
    change(ref)
    verdict = verify_task(D, terms, papers, tmp_path, ref)
    assert verdict["valid"] is False and message in verdict["message"]


@pytest.mark.parametrize(
    ("tool", "value", "valid"),
    [
        (Tool.EXACT_TOP_TERMS, D, True),
        (Tool.EXACT_TOP_TERMS, D | {"k": 0}, False),
        (Tool.EXACT_TOP_TERMS, D | {"k": 31}, False),
        (Tool.EXACT_TOP_TERMS, D | {"first_year": 2022}, False),
        (Tool.SELECT_PAPERS, {key: D[key] for key in ("dataset", "concept", "first_year", "last_year")}, True),
        (Tool.SELECT_PAPERS, {"dataset": "edge-cases", "concept": " ", "first_year": 2020, "last_year": 2021}, False),
        (Tool.RANK_TERMS, {"dataset": "edge-cases", "paper_set_id": "edge-cases-c0-2020-2021",
                           "idf_scope": "collection", "k": 5}, True),
        (Tool.RANK_TERMS, {"dataset": "edge-cases", "paper_set_id": "acl-small-c0-2020-2021",
                           "idf_scope": "collection", "k": 5}, False),
        (Tool.RANK_TERMS, {"dataset": "edge-cases", "paper_set_id": "edge-cases-c0-2020-2021",
                           "idf_scope": "selection", "k": 0}, False),
        (Tool.DESCRIBE_COLLECTION, {"dataset": "acl-small"}, True),
    ],
)
def test_checks_what_the_types_cannot_say(tool: Tool, value: dict, valid: bool) -> None:
    assert verify_input(tool, value)["valid"] is valid
