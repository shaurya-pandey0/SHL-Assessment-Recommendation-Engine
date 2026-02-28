"""
SHL Assessment Recommendation Engine — Streamlit Frontend

Provides a web UI for querying the recommendation API.
Supports both natural language queries and job description URLs.
"""

import os
import streamlit as st
import requests

# API URL — defaults to local, override with env var for deployment
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .assessment-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #1a73e8;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 4px;
    }
    .badge-yes { background: #e6f4ea; color: #1e8e3e; }
    .badge-no { background: #fce8e6; color: #d93025; }
    .badge-type { background: #e8f0fe; color: #1967d2; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🎯 SHL Assessment Recommendation Engine")
st.markdown(
    "Enter a **job description**, **natural language query**, or **URL** "
    "to find the most relevant SHL assessments."
)

# ── Input ────────────────────────────────────────────────────────────────────

query = st.text_area(
    "Enter your query or job description URL:",
    height=120,
    placeholder="e.g., 'Looking for a Python developer assessment under 30 minutes'\n"
                "or paste a job description URL..."
)

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button("🔍 Get Recommendations", type="primary")

# ── Results ──────────────────────────────────────────────────────────────────

if search_clicked and query.strip():
    with st.spinner("Finding relevant assessments..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query.strip()},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("recommended_assessments", [])

                if results:
                    st.success(f"Found {len(results)} relevant assessments")

                    for i, r in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### {i}. [{r['name']}]({r['url']})")

                            # Metadata row
                            cols = st.columns(4)
                            with cols[0]:
                                dur = f"{r['duration']} min" if r.get('duration') else "N/A"
                                st.metric("Duration", dur)
                            with cols[1]:
                                st.metric("Remote", r.get('remote_support', 'N/A'))
                            with cols[2]:
                                st.metric("Adaptive", r.get('adaptive_support', 'N/A'))
                            with cols[3]:
                                types = ", ".join(r.get('test_type', []))
                                st.metric("Type", types if len(types) < 25 else types[:22] + "...")

                            # Description
                            desc = r.get('description', '')
                            if desc:
                                preview = desc[:300] + "..." if len(desc) > 300 else desc
                                st.caption(preview)

                            st.divider()
                else:
                    st.warning("No assessments found for this query. Try a different query.")

            elif response.status_code == 400:
                st.error(f"Bad request: {response.json().get('detail', 'Unknown error')}")
            else:
                st.error(f"API error (status {response.status_code})")

        except requests.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the API. "
                f"Make sure the API is running at {API_URL}"
            )
        except requests.Timeout:
            st.error("⚠️ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif search_clicked:
    st.warning("Please enter a query.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "SHL Assessment Recommendation Engine | "
    "Built with Sentence-BERT + Google Gemini + FastAPI"
    "</div>",
    unsafe_allow_html=True,
)
