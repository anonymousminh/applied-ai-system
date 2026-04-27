"""Streamlit UI for VibeFinder 2.0.

Wraps the same `Orchestrator` pipeline used by `demo.py` in an interactive
front end so the end-to-end system can be demoed in a browser (and recorded
for the rubric's video walkthrough).

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from src.schemas import RecommendationResult


PROFILES_PATH = Path("examples/demo_profiles.json")
DIAGRAM_PATH = Path("assets/music-recommendation-diagram.png")
VALID_GENRES = ["pop", "lofi", "rock", "electronic", "folk", "jazz"]
VALID_MOODS = ["happy", "energetic", "calm", "melancholic", "intense"]


@st.cache_resource(show_spinner="Loading songs, building TF-IDF index…")
def get_orchestrator() -> Orchestrator:
    """Build the Orchestrator once per Streamlit session."""
    load_dotenv()
    return Orchestrator()


@st.cache_data
def load_demo_profiles() -> List[Dict[str, Any]]:
    if not PROFILES_PATH.exists():
        return []
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def render_profile_form(presets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sidebar form. Returns a raw profile dict ready for the orchestrator."""
    st.sidebar.header("1. Choose a profile")
    preset_labels = ["Custom"] + [p.get("name", f"Profile {i}") for i, p in enumerate(presets)]
    choice = st.sidebar.selectbox("Preset", preset_labels, index=1 if presets else 0)

    if choice == "Custom" or not presets:
        defaults = {
            "name": "Custom Profile",
            "favorite_genre": "pop",
            "favorite_mood": "energetic",
            "target_energy": 0.7,
            "likes_acoustic": False,
        }
    else:
        defaults = presets[preset_labels.index(choice) - 1]

    st.sidebar.header("2. Tune the profile")
    name = st.sidebar.text_input("Name", value=defaults.get("name", ""))

    genre_default = defaults.get("favorite_genre", "pop")
    genre_options = VALID_GENRES + (
        [genre_default] if genre_default not in VALID_GENRES else []
    )
    favorite_genre = st.sidebar.selectbox(
        "Favorite genre",
        genre_options,
        index=genre_options.index(genre_default),
        help="Out-of-catalog genres are allowed; the system will warn and fall back to mood + audio similarity.",
    )

    mood_default = defaults.get("favorite_mood", "energetic")
    mood_options = VALID_MOODS + (
        [mood_default] if mood_default not in VALID_MOODS else []
    )
    favorite_mood = st.sidebar.selectbox(
        "Favorite mood",
        mood_options,
        index=mood_options.index(mood_default),
    )

    target_energy = st.sidebar.slider(
        "Target energy",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults.get("target_energy", 0.7)),
        step=0.05,
    )
    likes_acoustic = st.sidebar.checkbox(
        "Likes acoustic", value=bool(defaults.get("likes_acoustic", False))
    )

    return {
        "name": name,
        "favorite_genre": favorite_genre,
        "favorite_mood": favorite_mood,
        "target_energy": target_energy,
        "likes_acoustic": likes_acoustic,
    }


def render_recommendation(idx: int, rec) -> None:
    grounded_badge = "Grounded" if rec.grounded else "Fallback"
    badge_color = "#16a34a" if rec.grounded else "#d97706"

    title_col, score_col, badge_col = st.columns([6, 2, 2])
    with title_col:
        st.markdown(f"### #{idx} {rec.song.title}")
        st.caption(f"by {rec.song.artist}")
    with score_col:
        st.metric("Score", f"{rec.score:.2f}")
    with badge_col:
        st.markdown(
            f"<div style='padding:6px 10px;border-radius:6px;"
            f"background:{badge_color};color:white;text-align:center;"
            f"font-weight:600;margin-top:18px'>{grounded_badge}</div>",
            unsafe_allow_html=True,
        )

    feat_cols = st.columns(5)
    feat_cols[0].caption(f"genre · **{rec.song.genre}**")
    feat_cols[1].caption(f"mood · **{rec.song.mood}**")
    feat_cols[2].caption(f"energy · **{rec.song.energy:.2f}**")
    feat_cols[3].caption(f"acoustic · **{rec.song.acousticness:.2f}**")
    feat_cols[4].caption(f"BPM · **{rec.song.tempo_bpm:.0f}**")

    st.write(rec.explanation)

    if rec.citations:
        with st.expander(f"Citations ({len(rec.citations)})"):
            for c in rec.citations:
                st.markdown(
                    f"- **{c.title}**  \n  "
                    f"`{c.passage_id}` · _{c.source_type}_"
                )

    with st.expander("Score breakdown"):
        st.json(rec.breakdown)

    st.divider()


def render_result(result: RecommendationResult) -> None:
    p = result.profile

    summary_cols = st.columns(4)
    summary_cols[0].metric("Genre", p.favorite_genre)
    summary_cols[1].metric("Mood", p.favorite_mood)
    summary_cols[2].metric("Target energy", f"{p.target_energy:.2f}")
    summary_cols[3].metric("Acoustic", "Yes" if p.likes_acoustic else "No")

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    if result.used_fallback:
        st.info(
            "At least one recommendation triggered the fallback guardrail "
            "(retrieval was weak or the explanation was not grounded)."
        )

    grounded_count = sum(1 for r in result.recommendations if r.grounded)
    st.caption(
        f"{grounded_count}/{len(result.recommendations)} recommendations are grounded in retrieved passages."
    )

    st.subheader("Recommendations")
    for i, rec in enumerate(result.recommendations, start=1):
        render_recommendation(i, rec)


def main() -> None:
    st.set_page_config(
        page_title="VibeFinder 2.0",
        page_icon="🎵",
        layout="wide",
    )

    st.title("VibeFinder 2.0")
    st.caption(
        "RAG-enhanced music recommender · rule-based scorer + retrieved context + groundedness guardrail"
    )

    presets = load_demo_profiles()
    raw_profile = render_profile_form(presets)

    st.sidebar.header("3. Pipeline settings")
    top_k = st.sidebar.slider("Recommendations (top_k)", 1, 5, 3)
    retrieve_k = st.sidebar.slider("Passages per song (retrieve_k)", 1, 5, 3)

    if DIAGRAM_PATH.exists():
        with st.sidebar.expander("Architecture diagram"):
            st.image(str(DIAGRAM_PATH), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Profile to send through the pipeline")
        st.json(raw_profile)
    with right:
        st.subheader("Run")
        run = st.button("Recommend songs", type="primary", use_container_width=True)
        st.caption(
            "Each click runs: validate → rule-score → retrieve passages → "
            "generate explanation → groundedness check."
        )

    if not run:
        st.info("Pick or tune a profile in the sidebar, then click **Recommend songs**.")
        return

    orch = get_orchestrator()
    try:
        with st.spinner("Running the full RAG pipeline…"):
            result = orch.recommend(raw_profile, top_k=top_k, retrieve_k=retrieve_k)
    except ValueError as e:
        st.error(f"Input rejected by guardrail: {e}")
        return

    render_result(result)


if __name__ == "__main__":
    main()
