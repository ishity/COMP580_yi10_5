"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    DESCRIBE_COLLECTION = "describe_collection"
    SELECT_PAPERS = "select_papers"
    RANK_TERMS = "rank_terms"
    EXACT_TOP_TERMS = "exact_top_terms"


class Task(Enum):
    TOP_TFIDF_TERMS = "top-tfidf-terms"


class Dataset(Enum):
    ACL_SMALL = "acl-small"
    EDGE_CASES = "edge-cases"


class IdfScope(Enum):
    COLLECTION = "collection"
    SELECTION = "selection"


DATASET = "The bundled collection of papers, each with its year, its concepts and its abstract"
CONCEPT = "A concept name of the dataset's vocabulary, exactly as written, e.g. question answering"
FIRST_YEAR = "The first publication year of the range, included"
LAST_YEAR = "The last publication year of the range, included"
PAPER_SET_ID = "The selection's id, as select_papers returned it"
IDF_SCOPE = (
    "The papers document frequencies are counted over: collection, every paper of the dataset, or selection,"
    " the selected papers only"
)
K = "How many top terms to return, from 1 to 30"
TERM = "A term: a normalized abstract token of at least 3 characters, not all digits and not a stopword"
TF = "How many times the term occurs in the selected papers' abstracts, summed over the papers"
DF = "How many papers of the scope contain the term at least once"
SCORE = "tf times ln(N / df), quantized to 6 decimal places (ranking uses the full precision)"
SCOPE_PAPERS = "N, how many papers the scope holds"


@dataclass
class RankedTerm(Output):
    term: Annotated[str, TERM]
    tf: Annotated[int, TF]
    df: Annotated[int, DF]
    score: Annotated[str, SCORE]


@dataclass
class DescribeInput(Output):
    dataset: Annotated[Dataset, DATASET]


@dataclass
class DescribeOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    papers: Annotated[int, "How many papers the dataset holds"]
    first_year: Annotated[int, "The earliest publication year of the dataset's papers"]
    last_year: Annotated[int, "The latest publication year of the dataset's papers"]
    terms: Annotated[int, "How many distinct terms the dataset's abstracts hold"]
    concepts: Annotated[int, "How many concepts the dataset's vocabulary holds"]
    concepts_with_papers: Annotated[int, "How many of those concepts at least one paper has"]


@dataclass
class SelectInput(Output):
    dataset: Annotated[Dataset, DATASET]
    concept: Annotated[str, CONCEPT]
    first_year: Annotated[int, FIRST_YEAR]
    last_year: Annotated[int, LAST_YEAR]


@dataclass
class SelectOutput(Output):
    paper_set_id: Annotated[str, "The selection's id, <dataset>-c<concept index>-<first_year>-<last_year>"]
    dataset: Annotated[Dataset, DATASET]
    concept: Annotated[str, "The concept the papers were selected by"]
    first_year: Annotated[int, FIRST_YEAR]
    last_year: Annotated[int, LAST_YEAR]
    papers: Annotated[int, "How many papers of the year range have the concept; 0 when none has"]


@dataclass
class RankInput(Output):
    dataset: Annotated[Dataset, DATASET]
    paper_set_id: Annotated[str, PAPER_SET_ID]
    idf_scope: Annotated[IdfScope, IDF_SCOPE]
    k: Annotated[int, K]


@dataclass
class RankOutput(Output):
    ranking_id: Annotated[str, "The saved ranking's id, <paper_set_id>-<idf_scope>"]
    paper_set_id: Annotated[str, PAPER_SET_ID]
    idf_scope: Annotated[IdfScope, "The scope the document frequencies were counted over"]
    selected_papers: Annotated[int, "How many papers the selection holds"]
    scope_papers: Annotated[int, SCOPE_PAPERS]
    ranked_terms: Annotated[int, "How many distinct terms the selected papers hold, all of them ranked"]
    top: Annotated[
        list[RankedTerm], "The first k terms, by score descending, then term ascending; all terms if fewer"
    ]


@dataclass
class TopTermsInput(Output):
    dataset: Annotated[Dataset, DATASET]
    concept: Annotated[str, "The topic: a concept name of the dataset's vocabulary"]
    first_year: Annotated[int, "The first publication year of the topic's papers, included"]
    last_year: Annotated[int, "The last publication year of the topic's papers, included"]
    k: Annotated[int, K]


@dataclass
class ExactOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    concept: Annotated[str, "The task's concept"]
    concept_index: Annotated[int, "The concept's 0-based index in the dataset's vocabulary, as paper_set_id writes it"]
    first_year: Annotated[int, FIRST_YEAR]
    last_year: Annotated[int, LAST_YEAR]
    k: Annotated[int, K]
    paper_ids: Annotated[list[str], "The anthology ids of the selected papers, in dataset order"]
    scope_papers: Annotated[int, "N, how many papers the whole collection holds, the correct scope"]
    ranked_terms: Annotated[int, "How many distinct terms the selected papers hold"]
    leaders: Annotated[
        list[RankedTerm],
        "The first k + 1 terms of the collection-scope ranking, or all terms if fewer: the top k and the runner-up",
    ]


@dataclass
class TopTermsOutput(Output):
    terms: Annotated[list[str], "The top k terms, best first, exactly as rank_terms returned them; empty if no paper"]
    papers: Annotated[int, "How many papers the topic's selection holds"]


TOOLS = {
    Tool.DESCRIBE_COLLECTION: (DescribeInput, DescribeOutput),
    Tool.SELECT_PAPERS: (SelectInput, SelectOutput),
    Tool.RANK_TERMS: (RankInput, RankOutput),
    Tool.EXACT_TOP_TERMS: (TopTermsInput, ExactOutput),
}
TASKS = {Task.TOP_TFIDF_TERMS: (TopTermsInput, TopTermsOutput)}
