# TF-IDF Keyword Extract

A NanoAgent plugin with one checkable job: rank the most distinctive terms of a research topic's papers by
TF-IDF. A topic is an NLP concept and a year range over a bundled collection of ACL Anthology abstracts. The
agent selects the topic's papers and ranks the terms of their abstracts, with document frequencies counted
over the whole collection, so that a word common across the field does not count as distinctive of the
topic. Tokens, stopwords, scores and ties follow fixed rules, and a deterministic verifier recomputes every
score. The plugin is a standard MCP server over stdio; NanoAgent owns its launch command, environment, input
and output checks and lifecycle.

## Tools

| Tool | Does | Code |
|---|---|---|
| `describe_collection` | Gives a dataset's paper count, its first and last year, its distinct terms, and its concepts with how many have a paper | `src/plugin/describe.py` |
| `select_papers` | Selects the papers of a year range whose concept set holds a concept, saves their anthology ids as `selections/<paper_set_id>.json` and returns the `paper_set_id` (`<dataset>-c<concept index>-<first_year>-<last_year>`), the criteria and the paper count; refuses an unknown concept, naming the closest valid ones, and years outside the dataset | `src/plugin/select.py` |
| `rank_terms` | Ranks every term of a saved selection with `idf_scope` `collection` (every paper of the dataset) or `selection` (the selected papers only), saves the full ranking as `rankings/<paper_set_id>-<idf_scope>.json` and returns the top `k` with tf, df, N and the displayed score | `src/plugin/rank.py` |
| `exact_top_terms` | Ranks the task's selection over the collection and carries the concept's index, its paper ids, N, the number of terms and the first k + 1 terms with their counts: the task's reference, which the model never sees | `src/plugin/exact.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `top-tfidf-terms` | Ranks the k most distinctive terms of a topic's papers, with document frequencies from the whole collection | `agent`, with `describe_collection`, `select_papers` and `rank_terms`; reference `exact_top_terms` |

The task input is a `dataset`, a `concept`, `first_year`, `last_year` and `k` (1 to 30). The output is
`terms`, the top k terms in order (fewer when the topic has fewer terms), and `papers`, the number of
selected papers. When no paper of the range has the concept, the answer is an empty list with `papers` 0.

### The exact rules

`src/plugin/rule.py` and `verifier/rule.py` hold the same file, word for word:

- **Tokens**: plugin 2's normalizer on the abstract: Unicode NFKC, lowercase, every character outside
  `a-z0-9` becomes a space (so `chart-parser` gives `chart` and `parser`), and a token longer than 3
  characters ending in `s`, but not `ss`, `us` or `is`, loses that `s` (`trees` becomes `tree`).
- **Terms**: normalized tokens of at least 3 characters that are not all digits and not stopwords.
- **Stopwords**: 165 English function words, written out in `rule.py` already normalized (`does` as `doe`).
- **Scope**: a set of papers; N is its size and df(t) the number of its papers containing t.
- **Score**: tf(t), the raw count of t summed over the selected papers, times ln(N / df(t)), computed with
  `decimal` at 40 significant digits: `Decimal(N) / Decimal(df)`, then `.ln()`, then times tf. A term in every
  paper of the scope scores exactly 0.
- **Ranking**: every term of the selection, by score descending, ties by term ascending; the top k are the
  first k, or all terms if fewer.
- **Display**: scores are shown quantized to 6 decimal places (ROUND_HALF_EVEN); ranking uses full precision.

The task's instructions state that document frequencies come from the whole collection. The tools compute
either scope, and choosing it is the model's judgment.

### Verification

`verify_task` checks, in order, and the first failure decides:

1. Malformed: no term is submitted twice, `papers` is not negative, and there are at most k terms. Score 0.
2. Paper count: `papers` equals the reference's number of selected papers. Score 0 otherwise.
3. Provenance: the verifier names the two files the correct pipeline writes, from the task input and the
   reference's concept index, and reads exactly those: `selections/<dataset>-c<index>-<first_year>-<last_year>.json`
   must hold the task's criteria and exactly the reference's paper ids, and, unless the selection holds no
   paper, `rankings/<paper_set_id>-collection.json` must agree with the reference (the same N, number of
   terms, and leading terms with their tf and df). For an empty selection, the selection alone is enough and
   no term may be submitted. No other workspace file is listed or read. Score 0 otherwise: a wrong scope or a
   wrong selection is the judgment the task measures.
4. Order: the submitted terms are an order-preserving subsequence of the reference's top k, recomputed from
   the reference's counts with the same `decimal` rule (the reference's k + 1 leaders must be in ranking
   order, which settles the boundary). A term outside the top k, or two terms out of order, scores 0.
5. Exactness: the list equals the reference's top k. A subsequence with terms left out scores
   |R ∩ T| / |R|, capped at (|R| − 1) / |R|.

The result is valid, with score 1.0, only when every check passes; an empty reference and an empty answer
are valid. Each failure message says what was wrong, for example the first rank that differs.

### Instances

| Dataset | Concept | Years | k | Papers | Top k | Selection scope | Off-by-one years |
|---|---|---|---|---|---|---|---|
| `acl-small` | question answering | 2018 to 2019 | 10 | 288 | question, answering, answer, reasoning, knowledge, reading, comprehension, visual, hop, dataset | 3 of 10 terms shared; score 0 | 2019 to 2019: 180 papers; score 0 |
| `acl-small` | named entity recognition | 2019 to 2019 | 15 | 84 | ner, entity, named, recognition, chemical, entitie, annotation, nested, embedding, relation, language, mention, tag, domain, word | 10 of 15 shared; score 0 | 2018 to 2019: 128 papers; score 0 |
| `acl-small` | masked language modeling | 2018 to 2018 | 5 | 0 | (empty) | the same empty answer, valid | 2018 to 2019: 6 papers; score 0 |
| `edge-cases` | parsing | 2020 to 2021 | 5 | 3 | parser, tree, parsing, chart, grammar | add, build, error, help, misread; score 0 | 2021 to 2021: 1 paper; score 0 |

In `edge-cases`, chart, grammar and lattice tie exactly at rank 4 to 6 (2 ln 5 = 3.218876), so the term
tie-break decides rank 5. Its 10 hand-written abstracts over 2020 and 2021 also hold a term in every paper
(`corpus`, score 0), stopwords and short tokens, plural forms, hyphenated words, all-digit tokens, a topic
with fewer than 30 terms, a concept whose papers are all in 2021, and a vocabulary concept with no paper.

### Data

`acl-small` holds the 2,924 papers of 2018 and 2019 that the shared gapmap corpus keeps (an abstract and at
least 2 of 493 NLP keywords matched as whole phrases in title and abstract), each with its anthology id,
year, concept indices into the 493 names, and its abstract as plain text. The concepts come from title and
abstract, while terms come from the abstract only. The generator lives outside this repository; only its
JSON output is here, from the ACL Anthology data repository at commit
`c7e7c44f91fba03d84e9fd7c57d693c4d6523e86`. Source: ACL Anthology, https://aclanthology.org/; the abstracts
are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), credited to their authors and
the ACL Anthology, and are reproduced unchanged.

## Layout

It has the standard plugin layout, which NanoAgent requires:

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # needed environment variable names (none here)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS table), one module per tool,
│                    # bundle.py (collections, selections, rankings and paths), rule.py (the exact rules)
│                    # and data/ (the JSON)
├── tests/           # test_tools.py and test_verifier.py
├── ui/              # renderers.js and renderers.css: outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, one module per tool, top.py for the task
```

`contract.py` types every value: the server builds each call's input into its class and checks its
output against its own, and NanoAgent and the verifier get the same classes. A failing tool raises,
and the MCP server returns an error result.

`verifier/` is deterministic: it imports only the standard library and `contract`, and reads only the
workspace files it names itself, with `Path(...).read_text()`: the call's own selection and ranking, and for
the task the task's selection and its collection-scope ranking. Beyond the task check, it rederives every tool call: for
`select_papers`, the id format and the saved selection's criteria and size; for `rank_terms`, the saved
ranking's selection, scope and N (equal to the selection's size for `selection`), that every term is a
normalized term with possible counts, that the order follows the recomputed scores, and that the returned
top k and its displayed scores are the ranking's first k; for `exact_top_terms`, that its leaders are in
ranking order with the rule's displayed scores.

## Develop and run

```bash
uv lock
uv sync --extra dev
uv run pytest
uv run ruff check .
uv lock --check
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`; `select_papers` writes
`selections/<paper_set_id>.json` there and `rank_terms` writes `rankings/<paper_set_id>-<idf_scope>.json`.
