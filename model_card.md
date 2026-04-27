# Model Card — VibeFinder 2.0

## 1. Model Name
VibeFinder 2.0 — RAG-enhanced music recommender.

## 2. Intended Use
Suggests 3–5 songs from a small catalog (`data/songs.csv`) given a user's
preferred genre, mood, target energy, and acoustic preference. Each
recommendation includes a grounded, cite-able explanation drawn from a
multi-source markdown knowledge base. **Classroom use only**, not a real
production recommender.

## 3. How It Works (Plain Language)
1. Validate the user profile against a strict schema; warn on unknown
   genres/moods.
2. Score every song with the original rule-based scorer (genre, mood,
   energy similarity, plus minor audio-feature contributions).
3. For each top candidate, build a query combining the profile and the
   song description.
4. Retrieve the most relevant passages from a TF-IDF index over three
   sources: artist bios, genre descriptions, and listener reviews.
5. Generate a 2–3 sentence explanation that combines (a) which feature
   matches drove the score and (b) up to two retrieved passages. Each
   referenced passage is cited by its passage ID.
6. Run a groundedness guardrail. If the explanation has no citations from
   actually-retrieved passages, replace it with a conservative
   feature-only fallback.

## 4. Data
- **Catalog:** 18 songs across 6 genres (pop, lofi, rock, electronic, folk,
  jazz). Author-written, designed to expose recognisable archetypes per
  genre/mood combination.
- **Knowledge base:**
  - `artist_bios.md` — one section per artist
  - `genre_descriptions.md` — one section per genre and per mood
  - `listener_reviews.md` — one section per song (synthetic listener notes)
- All data is author-written and reflects the author's prior class project's
  taste profile. It is small, English-only, and Western-pop-leaning.

## 5. Strengths
- **Transparent.** The rule-based scorer is fully readable and the RAG
  output is always cited.
- **Reproducible.** Default backend is deterministic; demo and evaluator do
  not require an API key.
- **Reliable failure modes.** Three independent guardrails (input validation,
  retrieval fallback, groundedness check) keep the system honest when input
  or retrieval is degenerate.
- **Multi-source retrieval** improves the perceived quality of explanations
  vs single-source retrieval.

## 6. Limitations and Bias
- Small catalog with limited intra-genre variety; certain artists dominate.
- Genre receives the largest weight in scoring, which can bias results
  against cross-genre matches that share mood and energy.
- TF-IDF retrieval is brittle on out-of-vocabulary queries (e.g., "kpop"
  has no term match in the corpus).
- The groundedness check verifies citation provenance, not per-sentence
  entailment, so a poorly written template could still cite a passage that
  doesn't quite support its claim.
- All knowledge-base text reflects one author's voice and taste.

## 7. Evaluation
- **Unit + integration tests:** `pytest` runs 21 tests covering scorer,
  retriever, guardrails, and end-to-end pipeline behavior.
- **Evaluation harness:** `python evaluator.py` runs all 5 demo profiles
  and reports precision@k (matches genre or mood), diversity@k (distinct
  artists), and groundedness (fraction of recommendations whose
  explanations cite at least one retrieved passage). Each profile receives
  a pass/fail verdict against configurable thresholds. The current
  thresholds are deliberately conservative; the system passes 5/5 on the
  default profile set.

## 8. Future Work
- Swap TF-IDF for a neural embedding retriever behind the same interface.
- Add a planning agent that adapts the retrieval query for contradictory
  profiles instead of concatenating profile and song descriptions.
- Per-sentence groundedness check.
- MMR-style diversity reranking.
- Expand both the catalog and the knowledge base.

## 9. Personal Reflection
See `reflection.md` for a full discussion of how AI was used, helpful and
flawed AI suggestions, and what I'd improve next.
