# Reflection — VibeFinder 2.0

## How I Used AI During Development

I used Cursor's agent and chat throughout this project, but in three distinct
modes:

1. **Design / planning conversations.** Before writing any code, I asked the
   AI to map the rubric to concrete deliverables and to draft the system
   architecture diagram. This forced an explicit mapping between rubric points
   and files in the repo, which made it much harder to leave a rubric line
   silently uncovered.
2. **Scaffolding the codebase.** Once the design was stable, I had the AI
   generate the initial module skeletons (`schemas.py`, `scorer.py`,
   `retriever.py`, `generator.py`, `guardrails.py`, `orchestrator.py`).
   Generating them in dependency order, with strict typing, kept the
   integration step short.
3. **Debugging and tightening.** When tests or the demo failed, I used the AI
   to diagnose specific issues (wrong types, off-by-one in slicing,
   inconsistent field names) rather than asking it to "make it work."

## One Helpful AI Suggestion

The AI pushed me toward a **dual-backend generator** — a deterministic
template generator as the default, with an optional OpenAI generator behind
an env-var switch and a graceful fallback. This was helpful for two reasons:

- The demo and evaluator are fully reproducible without an API key, which
  matters for grading and for anyone who clones the repo.
- The OpenAI path becomes a real upgrade rather than a load-bearing
  dependency. If the LLM call fails, the template generator absorbs the
  failure and the user still gets a grounded explanation with citations.

I did not anticipate this design at the start; my first instinct was
"plug in the LLM directly." The AI's framing of "make the LLM optional and
keep the contract identical" turned out to be the right call.

## One Flawed AI Suggestion

The AI initially proposed using **sentence-transformers** for retrieval and
suggested a cosine similarity threshold of `0.5` for filtering low-quality
matches. Both were wrong for this project:

- Sentence-transformers requires a ~500MB model download on first run, which
  hurts reproducibility and runs counter to the "lightweight class project"
  framing.
- The `0.5` threshold was far too aggressive. With short queries and
  relatively short passages, cosine similarity scores using TF-IDF rarely
  exceed `0.3` even when the retrieval is clearly relevant. A threshold that
  high would have caused the retrieval-fallback guardrail to trigger
  constantly, suppressing useful context.

I switched to TF-IDF + cosine and lowered the usability threshold to `0.05`,
which the demo confirms is more honest: most retrievals land in the
`0.1–0.4` range, and the threshold meaningfully separates "no signal" from
"some relevant context."

The general lesson: AI suggestions for thresholds, weights, and tolerances
should never be accepted without empirical confirmation on real data.

## System Limitations

- **Small catalog.** With 18 songs across 6 genres, certain artists (Soft
  Static, Iron Echo) dominate within their genres. Diversity at top-3 is
  bounded by the data, not the algorithm.
- **TF-IDF over short passages.** TF-IDF works well here but is brittle when
  queries use words that don't appear in the corpus (e.g., a query for
  "kpop" gets no exact term match and falls back to mood/energy retrieval).
- **Genre weight dominance.** The rule-based scorer still weighs genre at
  `+2.0`, the largest fixed weight. For users with contradictory profiles
  (high energy + melancholic), the top result is dragged toward the user's
  genre even when it's a poor mood match.
- **Groundedness check is shallow.** The current check verifies that
  citations were taken from the retrieved set, not that every sentence in
  the explanation is supported by those passages. A stronger version would
  do per-sentence entailment.
- **No real user feedback.** All "user profiles" are synthetic.

## Future Improvements

1. **Swap retriever to a neural embedding model** behind the same interface
   so the rest of the pipeline stays unchanged. The TF-IDF retriever is
   `Retriever.build(passages)` returning an object with a single
   `retrieve(query, top_k, ...)` method, so swapping is one class away.
2. **Add an agentic planning step** in front of retrieval — for contradictory
   profiles, plan a "bridge" query (e.g., "high-energy melancholic pop")
   instead of using the naive concatenation.
3. **Per-sentence groundedness** — split the explanation into sentences and
   require each to map to at least one citation, falling back to the
   conservative explanation if any sentence is unsupported.
4. **Diversity-aware reranking.** Apply maximal marginal relevance to the
   top-k candidates so the same artist isn't repeated unnecessarily.
5. **Expand the catalog and the knowledge base** — at minimum 50 songs and
   one passage per song to reduce the dominance of a few well-described
   tracks in retrieval results.

## What Surprised Me

The biggest surprise was how quickly the **citation requirement** improved
the perceived quality of the system. Even with a deterministic
template-based generator, the explanations feel honest and verifiable
because every claim is tied to a passage ID the user can trace back to a
markdown file. This shifted my mental model of RAG: the value is less about
"the LLM wrote prettier text" and more about "every claim is checkable."
