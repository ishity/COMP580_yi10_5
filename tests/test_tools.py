from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import TOOLS, Dataset, DescribeInput, IdfScope, RankInput, SelectInput, Tool, TopTermsInput
from plugin import describe_collection, exact_top_terms, rank_terms, rule, select_papers

ROOT = Path(__file__).resolve().parents[1]
EDGE = Dataset.EDGE_CASES
SMALL = Dataset.ACL_SMALL


def checked(tool: Tool, given, made, workspace: Path) -> dict:
    """verify_output on a tool call, with values built the way the sandbox builds them."""
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(
        tool.value,
        build_output(plain_output(given), given_class),
        build_output(plain_output(made), made_class),
        str(workspace),
    )


def select(workspace: Path, dataset: Dataset, concept: str, first: int, last: int):
    given = SelectInput(dataset, concept, first, last)
    return given, select_papers(given, workspace)


def rank(workspace: Path, dataset: Dataset, paper_set_id: str, scope: IdfScope, k: int):
    given = RankInput(dataset, paper_set_id, scope, k)
    return given, rank_terms(given, workspace)


def ranked(workspace: Path, dataset: Dataset, concept: str, first: int, last: int, scope: IdfScope, k: int):
    _, chosen = select(workspace, dataset, concept, first, last)
    return rank(workspace, dataset, chosen.paper_set_id, scope, k)[1]


@pytest.mark.parametrize(
    ("dataset", "sizes"),
    [(SMALL, (2924, 2018, 2019, 10204, 493, 279)), (EDGE, (10, 2020, 2021, 36, 4, 3))],
)
def test_describe_collection_counts_the_bundle(tmp_path: Path, dataset: Dataset, sizes: tuple) -> None:
    given = DescribeInput(dataset)
    made = describe_collection(given, tmp_path)
    assert (made.papers, made.first_year, made.last_year, made.terms, made.concepts, made.concepts_with_papers) == sizes
    assert checked(Tool.DESCRIBE_COLLECTION, given, made, tmp_path)["valid"] is True


def test_the_two_rule_files_are_identical() -> None:
    assert (ROOT / "src/plugin/rule.py").read_text() == (ROOT / "verifier/rule.py").read_text()


def test_stopwords_are_normalized_terms_of_three_characters_or_more() -> None:
    assert 140 <= len(rule.STOPWORDS) <= 170
    for word in rule.STOPWORDS:
        assert rule.normalize(word) == [word] and len(word) >= rule.MIN_LENGTH


def test_tokenization_rules() -> None:
    # Stopwords and short tokens dropped.
    assert rule.terms("Is it of an ab xy help? The parser and the grammar") == ["help", "parser", "grammar"]
    # Plurals merge into one term; "ss", "us", "is" and short words keep their "s".
    assert rule.terms("trees tree models class corpus analysis gas") == [
        "tree", "tree", "model", "class", "corpus", "analysis", "gas"
    ]
    # Hyphenated words split; all-digit tokens dropped, mixed ones kept.
    assert rule.terms("a chart-parser, shift-reduce; 2020 and 100 bert2") == [
        "chart", "parser", "shift", "reduce", "bert2"
    ]
    # NFKC folds compatibility characters before lowercasing.
    assert rule.terms("ﬁne-tuning Ｍodels") == ["fine", "tuning", "model"]


def test_the_score_rule() -> None:
    assert rule.score(3, 10, 10) == 0
    assert rule.display(rule.score(2, 2, 10)) == "3.218876"  # 2 ln 5
    assert rule.score(4, 3, 10) == rule.score(4, 3, 10) and rule.display(rule.score(1, 1, 10)) == "2.302585"
    assert math.isclose(float(rule.score(7, 3, 2924)), 7 * math.log(2924 / 3), rel_tol=1e-15)
    with pytest.raises(ValueError):
        rule.score(1, 0, 10)
    assert rule.display(rule.score(1, 1, 1)) == "0.000000"


def test_a_term_in_every_paper_of_the_scope_scores_zero(tmp_path: Path) -> None:
    made = ranked(tmp_path, EDGE, "parsing", 2020, 2021, IdfScope.COLLECTION, 30)
    corpus = [term for term in made.top if term.term == "corpus"]
    assert corpus and (corpus[0].tf, corpus[0].df, made.scope_papers, corpus[0].score) == (3, 10, 10, "0.000000")
    assert made.top[-1].term == "corpus"


def test_an_exact_tie_is_broken_by_term_order(tmp_path: Path) -> None:
    made = ranked(tmp_path, EDGE, "parsing", 2020, 2021, IdfScope.COLLECTION, 6)
    assert [(t.term, t.tf, t.df, t.score) for t in made.top] == [
        ("parser", 4, 3, "4.815891"), ("tree", 4, 3, "4.815891"), ("parsing", 3, 3, "3.611918"),
        ("chart", 2, 2, "3.218876"), ("grammar", 2, 2, "3.218876"), ("lattice", 2, 2, "3.218876"),
    ]


def test_k_larger_than_the_terms_returns_them_all(tmp_path: Path) -> None:
    made = ranked(tmp_path, EDGE, "parsing", 2020, 2021, IdfScope.COLLECTION, 30)
    assert made.ranked_terms == 18 and len(made.top) == 18


def test_the_selection_scope_changes_the_top_k(tmp_path: Path) -> None:
    right = ranked(tmp_path, EDGE, "parsing", 2020, 2021, IdfScope.COLLECTION, 5)
    wrong = ranked(tmp_path, EDGE, "parsing", 2020, 2021, IdfScope.SELECTION, 5)
    assert [t.term for t in right.top] == ["parser", "tree", "parsing", "chart", "grammar"]
    assert [t.term for t in wrong.top] == ["add", "build", "error", "help", "misread"]
    assert (right.scope_papers, wrong.scope_papers) == (10, 3)
    assert (tmp_path / "rankings" / "edge-cases-c0-2020-2021-selection.json").is_file()


@pytest.mark.parametrize(
    ("dataset", "concept", "years", "papers"),
    [(EDGE, "parsing", (2020, 2021), 3), (EDGE, "parsing", (2020, 2020), 2), (EDGE, "parsing", (2021, 2021), 1),
     (SMALL, "question answering", (2018, 2019), 288), (SMALL, "question answering", (2019, 2019), 180),
     (SMALL, "named entity recognition", (2019, 2019), 84), (SMALL, "named entity recognition", (2018, 2019), 128),
     (SMALL, "masked language modeling", (2018, 2018), 0), (SMALL, "masked language modeling", (2018, 2019), 6)],
)
def test_a_year_range_off_by_one_changes_the_paper_count(
    tmp_path: Path, dataset: Dataset, concept: str, years: tuple, papers: int
) -> None:
    given, made = select(tmp_path, dataset, concept, *years)
    assert made.papers == papers
    assert checked(Tool.SELECT_PAPERS, given, made, tmp_path)["valid"] is True


def test_a_concept_with_no_paper_in_the_range_ranks_nothing(tmp_path: Path) -> None:
    _, chosen = select(tmp_path, EDGE, "summarization", 2020, 2020)
    assert chosen.papers == 0
    given, made = rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.COLLECTION, 5)
    assert (made.top, made.ranked_terms, made.selected_papers, made.scope_papers) == ([], 0, 0, 10)
    assert checked(Tool.RANK_TERMS, given, made, tmp_path)["valid"] is True
    assert select(tmp_path, EDGE, "speech recognition", 2020, 2021)[1].papers == 0  # no paper in any year


def test_a_misspelled_concept_is_refused_with_the_closest_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'question answring' is not a concept of acl-small") as error:
        select(tmp_path, SMALL, "question answring", 2018, 2019)
    assert "'question answering'" in str(error.value)
    assert not (tmp_path / "selections").exists()


@pytest.mark.parametrize(
    ("years", "message"),
    [((2017, 2019), "within acl-small's years 2018 to 2019"), ((2019, 2020), "2018 <= first_year"),
     ((2019, 2018), "years 2019 to 2018 are not a range")],
)
def test_select_papers_refuses_years_outside_the_collection(tmp_path: Path, years: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select(tmp_path, SMALL, "question answering", *years)


@pytest.mark.parametrize(
    ("paper_set_id", "k", "message"),
    [("edge-cases-c0-2020-2021", 0, "k must be from 1 to 30"), ("edge-cases-c0-2020-2021", 31, "got 31"),
     ("acl-small-c0-2020-2021", 5, "is not a edge-cases selection"), ("edge-cases-c1-2020-2021", 5, "call select_papers")],
)
def test_rank_terms_says_what_is_wrong(tmp_path: Path, paper_set_id: str, k: int, message: str) -> None:
    select(tmp_path, EDGE, "parsing", 2020, 2021)
    with pytest.raises(ValueError, match=message):
        rank(tmp_path, EDGE, paper_set_id, IdfScope.COLLECTION, k)


def test_rank_terms_refuses_a_changed_selection(tmp_path: Path) -> None:
    _, chosen = select(tmp_path, EDGE, "parsing", 2020, 2021)
    path = tmp_path / "selections" / f"{chosen.paper_set_id}.json"
    saved = json.loads(path.read_text())
    saved["paper_ids"].pop()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="was changed"):
        rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.COLLECTION, 5)


def test_the_saved_ranking_remembers_the_most_terms_returned(tmp_path: Path) -> None:
    _, chosen = select(tmp_path, EDGE, "parsing", 2020, 2021)
    path = tmp_path / "rankings" / "edge-cases-c0-2020-2021-collection.json"
    for k, returned in ((10, 10), (5, 10), (12, 12)):
        rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.COLLECTION, k)
        assert json.loads(path.read_text())["returned"] == returned


def test_rankings_are_deterministic(tmp_path: Path) -> None:
    for folder in ("one", "two"):
        ranked(tmp_path / folder, SMALL, "question answering", 2018, 2019, IdfScope.COLLECTION, 10)
    name = "rankings/acl-small-c43-2018-2019-collection.json"
    assert (tmp_path / "one" / name).read_text() == (tmp_path / "two" / name).read_text()


def slow_top(dataset: Dataset, concept: str, first: int, last: int, k: int) -> list[str]:
    """A second computation straight from the raw JSON, in floats: fine where no two scores are within 1e-9."""
    raw = json.loads((ROOT / "src/plugin/data" / f"{dataset.value}.json").read_text())
    index = raw["concepts"].index(concept)
    bags = [Counter(rule.terms(paper["abstract"])) for paper in raw["papers"]]
    df = Counter(term for bag in bags for term in bag)
    tf: Counter = Counter()
    for paper, bag in zip(raw["papers"], bags):
        if index in paper["concepts"] and first <= paper["year"] <= last:
            tf += bag
    scores = sorted(((-count * math.log(len(bags) / df[term]), term) for term, count in tf.items()))
    return [term for _, term in scores[:k]]


@pytest.mark.parametrize(
    ("concept", "years", "k"), [("question answering", (2018, 2019), 10), ("named entity recognition", (2019, 2019), 15)]
)
def test_the_reference_matches_a_slow_computation(tmp_path: Path, concept: str, years: tuple, k: int) -> None:
    made = exact_top_terms(TopTermsInput(SMALL, concept, *years, k), tmp_path)
    assert [t.term for t in made.leaders[:k]] == slow_top(SMALL, concept, *years, k)
    assert len(made.leaders) == k + 1
    assert f"-c{made.concept_index}-" in select(tmp_path, SMALL, concept, *years)[1].paper_set_id
    assert [t.term for t in made.leaders] == [
        t.term for t in ranked(tmp_path, SMALL, concept, *years, IdfScope.COLLECTION, k + 1).top
    ]


def test_verify_output_rederives_each_call(tmp_path: Path) -> None:
    given_select, chosen = select(tmp_path, SMALL, "question answering", 2018, 2019)
    assert checked(Tool.SELECT_PAPERS, given_select, chosen, tmp_path)["valid"] is True
    for scope in IdfScope:
        given, made = rank(tmp_path, SMALL, chosen.paper_set_id, scope, 10)
        assert checked(Tool.RANK_TERMS, given, made, tmp_path)["valid"] is True
    task = TopTermsInput(SMALL, "question answering", 2018, 2019, 10)
    assert checked(Tool.EXACT_TOP_TERMS, task, exact_top_terms(task, tmp_path), tmp_path)["valid"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda saved: saved["terms"].reverse(), "not ranked by score"),
        (lambda saved: saved["terms"][0].__setitem__(2, 99), "impossible counts tf 4, df 99 for N 10"),
        (lambda saved: saved["terms"][0].__setitem__(1, 1), "not ranked by score"),
        (lambda saved: saved.update(scope_papers=9), "N is 9"),
        (lambda saved: saved.update(returned=3), "fewer than k"),
        (lambda saved: saved["terms"][-1].__setitem__(0, "the"), "'the' is not a normalized term"),
    ],
    ids=["order", "df", "tf", "n", "returned", "stopword"],
)
def test_verify_output_refuses_a_tampered_ranking(tmp_path: Path, change, message: str) -> None:
    _, chosen = select(tmp_path, EDGE, "parsing", 2020, 2021)
    given, made = rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.COLLECTION, 5)
    path = tmp_path / "rankings" / f"{made.ranking_id}.json"
    saved = json.loads(path.read_text())
    change(saved)
    path.write_text(json.dumps(saved))
    verdict = checked(Tool.RANK_TERMS, given, made, tmp_path)
    assert verdict["valid"] is False and message in verdict["message"]


def test_verify_output_refuses_a_shown_score_off_the_rule(tmp_path: Path) -> None:
    _, chosen = select(tmp_path, EDGE, "parsing", 2020, 2021)
    given, made = rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.COLLECTION, 5)
    made.top[0].score = "4.815890"
    verdict = checked(Tool.RANK_TERMS, given, made, tmp_path)
    assert verdict["valid"] is False and "with their scores" in verdict["message"]


def test_verify_output_refuses_a_selection_scope_with_another_n(tmp_path: Path) -> None:
    _, chosen = select(tmp_path, EDGE, "parsing", 2020, 2021)
    given, made = rank(tmp_path, EDGE, chosen.paper_set_id, IdfScope.SELECTION, 5)
    path = tmp_path / "rankings" / f"{made.ranking_id}.json"
    saved = json.loads(path.read_text())
    saved["scope_papers"] = 10
    path.write_text(json.dumps(saved))
    made.scope_papers = 10
    verdict = checked(Tool.RANK_TERMS, given, made, tmp_path)
    assert verdict["valid"] is False and "cannot hold N = 10" in verdict["message"]


def test_verify_output_refuses_a_selection_of_another_size(tmp_path: Path) -> None:
    given, made = select(tmp_path, EDGE, "parsing", 2020, 2021)
    made.papers = 2
    verdict = checked(Tool.SELECT_PAPERS, given, made, tmp_path)
    assert verdict["valid"] is False and "not 2" in verdict["message"]


def test_edge_cases_hold_each_situation_by_construction() -> None:
    raw = json.loads((ROOT / "src/plugin/data/edge-cases.json").read_text())
    text = " ".join(paper["abstract"] for paper in raw["papers"])
    for word in ("chart-parser", "shift-reduce", "2020", "100", "trees", " tree ", "Is it of an ab xy"):
        assert word in text
    assert {paper["year"] for paper in raw["papers"]} == {2020, 2021}
    assert all("corpus" in rule.terms(paper["abstract"]) for paper in raw["papers"])
    years = {paper["year"] for paper in raw["papers"] if 2 in paper["concepts"]}  # summarization
    assert years == {2021}
