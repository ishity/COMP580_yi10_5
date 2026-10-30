# CLAUDE.md: build the `tfidf-keyword-extract` NanoAgent plugin

## Context

This is plugin 5 of 5 for the COMP 480/580 milestone "Verifiable Tasks" (due 2026-11-02). Plugins 1 to 3 are finished; plugin 4 (`gap-backtest`, `~/COMP580_yi10_4`) is in progress in parallel, so do not copy from it or touch it.

- Plugin 2, `concept-cooccurrence-query` (`~/COMP580_yi10_2`): the shared ACL Anthology corpus and its text normalizer.
- Plugin 3, `concept-gap-detection` (`~/COMP580_yi10_3`): the most recent finished plugin; copy its structure.

Goal: an original task, **ranking the most distinctive terms of a topic's papers by TF-IDF**, under fixed and documented tokenization, stopword, scoring and tie-breaking rules. A deterministic verifier recomputes every score.

**This plugin is deliberately light, like plugin 4.** There is no sketch and no budget. The model's judgment is in building the pipeline correctly: selecting the right papers and choosing the right scope for document frequencies. The tools can compute IDF over either the whole collection or the selected papers only, and the task defines which one is correct; the verifier checks that the right one was used.

## Read first

1. `~/nanoagent/docs/instructions/skills/publish-plugin/SKILL.md`.
2. `~/COMP580_yi10_3` in full, and `~/COMP580_yi10_2/src/plugin/` for how concepts were matched.
3. `~/gapmap-corpus-gen/README.md` for the corpus, the pinned ACL Anthology commit and the normalizer.

Patterns that still apply: ids instead of long item lists, informative errors stating the valid value, the reference tool taking the full task input, the verifier rederiving every call from workspace files, `rule.py` identical word for word in `src/plugin/` and `verifier/`.

If anything here conflicts with SKILL.md, SKILL.md wins. Tell me about the conflict.

## Hard rules

- **uv only.** Keep `requires-python`, dependencies and the `slimconfig` source unchanged except for the package-data setting.
- **Verifier**: pure standard library plus `contract`; no LLM, clock, randomness, environment, network, subprocess, `open`, `eval`, `exec`, `getattr`, private attributes or writes. `Path(...).read_text()` only. Deterministic across two runs.
- **Rules in code, judgment to the model. No fallbacks. Only allowed top-level files.**
- **Never run `git push` or `gh repo create`, never make anything public, and do not commit unless I say so.**
- No em-dashes in any text you write.

## Setup

```bash
cd ~
rsync -a --exclude .git --exclude .venv --exclude CLAUDE.md COMP580_yi10_3/ COMP580_yi10_5/
cd COMP580_yi10_5
git init -b main
```

This file becomes the repo's `CLAUDE.md`. Rename everywhere (`nanoagent.yaml` id `tfidf-keyword-extract`, name "TF-IDF Keyword Extract", README title, `pyproject.toml` name `nanoagent-tfidf-keyword-extract`), version 0.1.0, `status: in-development`, LICENSE stays "Yuri", 2026. Remove every MinHash and gap tool, task, test, verifier module and data file.

## Data

Add `~/gapmap-corpus-gen/derive_tfidf.py`. Using the same pinned ACL Anthology commit, it writes this plugin's JSON into `src/plugin/data/`:

- `acl-small`: the same 2,924 papers (2018 and 2019) as plugins 2 and 3, each with its anthology id, year, abstract text, and its concept set (keyword names, as matched in the shared corpus). Keep the file size reasonable; report it at Checkpoint 1.
- `edge-cases`: about 10 hand-written short abstracts over 2 years, see Tests.

Abstracts are CC BY 4.0; credit the ACL Anthology in the README.

## Exact rules (`rule.py`, identical in both places)

- **Tokens**: apply plugin 2's normalizer to the abstract (NFKC, lowercase, every character outside a-z0-9 becomes a space, drop a final `s` from tokens longer than 3 characters unless they end in `ss`, `us` or `is`). Keep tokens of at least 3 characters that are not all digits and not in the stopword list.
- **Stopwords**: a fixed English list written out as a constant in `rule.py` (about 150 common function words, already normalized by the same rule). No external stopword package.
- **Scope**: a scope is a set of papers. N is its size and df(t) is how many of its papers contain term t.
- **Score** of term t for a selected paper set S under a scope: `sum over papers d in S of tf(t, d)`, times `ln(N / df(t))`, where tf is the raw count. Compute with `decimal` at a fixed context precision (for example 40 digits): `Decimal(N) / Decimal(df)` then `.ln()`. Every paper in S must belong to the scope.
- **Ranking**: every term that occurs in S, by score descending, ties by term ascending. Top k is the first k, or all terms if fewer.
- **Display**: scores shown to the model are quantized to 6 decimal places (ROUND_HALF_EVEN); ranking always uses the full-precision value.

The verifier cannot read the abstracts, so the reference output must carry, for every term of the correct answer and for enough runners-up to decide the boundary, its summed tf, df and N. Propose exactly what the reference carries at Checkpoint 3.

## Tools

| Tool | Input | Output | Notes |
|---|---|---|---|
| `describe_collection` | `dataset` | paper count, years, distinct term count, concept count | |
| `select_papers` | `dataset`, `concept`, `first_year`, `last_year` | `paper_set_id`, the selection criteria echoed, paper count | Papers in the year range whose concept set contains `concept`. Raises on an unknown concept with the closest valid names, or on years outside the collection. Saves the id list to `workspace/selections/<paper_set_id>.json`. |
| `rank_terms` | `dataset`, `paper_set_id`, `idf_scope` (enum: `collection` or `selection`), `k` | the top k terms with summed tf, df, N and the displayed score; the scope used | `collection` uses all papers of the dataset as the scope; `selection` uses the selected papers only. Saves the full ranking to `workspace/rankings/<id>.json`. |
| `exact_top_terms` | full task input | what the verifier needs (see above) | **Reference only.** Not in the task's `tools`. |

## Task

- id `top-tfidf-terms`, runner `agent`, tools `[describe_collection, select_papers, rank_terms]`, reference `exact_top_terms`.
- Input: `dataset`, `concept`, `first_year`, `last_year`, `k` (1 to 30).
- Output: `terms` (an ordered list of k terms, or fewer) and `papers` (how many papers were selected).
- Instructions: define the selection, the tokenization in words, and the score precisely, and **state that document frequencies come from the whole collection**, because a term common across the field should not count as distinctive of the topic. State that an empty `terms` list with `papers` 0 is correct when no paper matches. Do not prescribe the exact call sequence.

## Verifier

`verify_task`, in order, first failure decides:
1. Malformed: no duplicate terms; `papers` ≥ 0; at most k terms.
2. Paper count equals the reference.
3. Provenance: the submitted list equals the top of a `rank_terms` result whose saved ranking used `idf_scope` `collection` on a selection matching the task's concept and years.
4. Exactness: the list equals the reference ranking's top k, recomputed by the verifier from the reference's counts with the same `decimal` rule.
5. Valid only when every check passes. Score 1.0 when valid; otherwise propose a partial score (for example the share of the reference's top k that appears in the submission, and 0 if the paper count is wrong) at Checkpoint 2.

`verify_output` rederives each call from the saved selection and ranking files where possible.

## Tests

Mirror the earlier plugins, plus: a term in every paper of the scope (ln 1 = 0, score 0); an exact score tie broken by term order; stopwords and short tokens dropped; plural normalization merging two forms; hyphenated words split; an all-digit token dropped; `k` larger than the number of terms; a concept with no paper in the range (empty answer); `selection` scope giving a different top k from `collection` (so the wrong scope fails); a year range off by one giving a different paper count. `edge-cases` must contain each situation by construction.

## Work plan and checkpoints

Stop at each checkpoint and wait for my OK.

1. Setup, rename, derive data. **Checkpoint 1**: data file sizes, term counts, and for a few concepts (for example `low-resource languages`, `question answering`, `data augmentation`) the top 10 terms under each scope for 2018 to 2019, so we can see that the scopes differ and the `collection` terms look distinctive.
2. **Checkpoint 2**: proposed instances (two main ones on `acl-small` with different concepts and k, one with no matching papers, and edge-cases), showing for each that the wrong scope and an off-by-one year range change the answer; plus the partial score proposal.
3. `contract.py`. **Checkpoint 3**: show it, with what the reference carries.
4. Tools, manifest, verifier, tests.
5. UI, product page, README (`## Tools`, `## Layout`, `## Develop and run`, the instance table, the exact rules, the verifier order, the data source and license).
6. All checks: `uv lock`, `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`, `uv lock --check`, `nanoagent plugin inspect .`, SKILL.md section 8. Status `ready`. Run a scripted agent end to end through the real MCP server and sandboxed verifier, with the correct pipeline, the wrong scope, and an off-by-one year range. **Checkpoint 4**: results, plus the exact instruction and tool text the model will see.

## Report notes

Keep them in `~/tfidf-report-notes.md`, never in this repo: derivation command, file sizes, chosen instances with reasons, the scope comparison table, every verifier case and verdict, test output, and limitations (for example: unigrams only; a naive plural rule; a fixed small stopword list; topic membership comes from exact phrase matching of concepts; TF-IDF is one of many term-weighting schemes).