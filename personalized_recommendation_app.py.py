import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from pathlib import Path


st.set_page_config(
    page_title="Personalized Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)


st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f7f5ff 0%,
            #eef5ff 50%,
            #f8f9ff 100%
        );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #24134f 0%,
            #34206f 55%,
            #172554 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #6d28d9,
            #2563eb,
            #0891b2
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        line-height: 1.2;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 32px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #1e293b;
        margin-top: 28px;
        margin-bottom: 16px;
    }

    .info-card {
        background: white;
        padding: 22px 15px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(
            30,
            41,
            59,
            0.08
        );
        border: 1px solid #e2e8f0;
        text-align: center;
        min-height: 105px;
    }

    .info-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 650;
    }

    .info-value {
        color: #312e81;
        font-size: 24px;
        font-weight: 800;
        margin-top: 7px;
    }

    .recommend-card {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f5f3ff 100%
        );
        border: 1px solid #ddd6fe;
        border-radius: 17px;
        padding: 19px;
        margin-bottom: 12px;
        box-shadow: 0 7px 20px rgba(
            79,
            70,
            229,
            0.08
        );
    }

    .recommend-rank {
        color: #7c3aed;
        font-size: 14px;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .recommend-item {
        color: #1e293b;
        font-size: 19px;
        font-weight: 750;
    }

    .method-card {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #ecfeff 100%
        );
        border-radius: 16px;
        padding: 17px 20px;
        border-left: 5px solid #6366f1;
        margin-bottom: 20px;
        color: #334155;
        box-shadow: 0 5px 16px rgba(
            79,
            70,
            229,
            0.06
        );
    }

    .method-name {
        color: #4338ca;
        font-weight: 800;
    }

    .segment-card {
        background: white;
        border-radius: 16px;
        padding: 18px 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 18px rgba(
            30,
            41,
            59,
            0.06
        );
    }

    .segment-name {
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
    }

    .segment-value {
        color: #4f46e5;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 35px 0 10px 0;
        font-size: 14px;
    }

    .stTextInput label {
        color: #334155 !important;
        font-weight: 700 !important;
    }

    .stTextInput input {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load saved files

BASE_DIR = Path(__file__).resolve().parent

if (BASE_DIR / "models").exists():
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
else:
    MODEL_DIR = BASE_DIR.parent / "models"
    RESULTS_DIR = BASE_DIR.parent / "results"


required_files = [
    MODEL_DIR / "recommendation_svd_model.pkl",
    MODEL_DIR / "recommendation_user_mapping.pkl",
    MODEL_DIR / "recommendation_item_mapping.pkl",
    MODEL_DIR / "recommendation_index_to_item.pkl",
    MODEL_DIR / "recommendation_popular_items.pkl",
    MODEL_DIR / "recommendation_user_history.pkl",
    RESULTS_DIR / "recommendation_model_comparison.csv",
    RESULTS_DIR / "user_segments.csv"
]


missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]


if missing_files:

    st.error(
        "Required model or result files were not found."
    )

    st.write("Expected model folder:")
    st.code(str(MODEL_DIR))

    st.write("Expected results folder:")
    st.code(str(RESULTS_DIR))

    st.write("Missing files:")

    for file in missing_files:
        st.write(file)

    st.stop()


svd_model = joblib.load(
    MODEL_DIR / "recommendation_svd_model.pkl"
)

user_to_index = joblib.load(
    MODEL_DIR / "recommendation_user_mapping.pkl"
)

item_to_index = joblib.load(
    MODEL_DIR / "recommendation_item_mapping.pkl"
)

index_to_item = joblib.load(
    MODEL_DIR / "recommendation_index_to_item.pkl"
)

popular_items = joblib.load(
    MODEL_DIR / "recommendation_popular_items.pkl"
)

user_history = joblib.load(
    MODEL_DIR / "recommendation_user_history.pkl"
)

model_comparison = pd.read_csv(
    RESULTS_DIR / "recommendation_model_comparison.csv"
)

user_segments = pd.read_csv(
    RESULTS_DIR / "user_segments.csv"
)


# Recommendation function

def get_recommendations(
    visitor_id,
    n_recommendations=10
):

    if visitor_id not in user_to_index:

        return (
            list(
                popular_items[
                    :n_recommendations
                ]
            ),
            "Popularity Fallback"
        )


    history = user_history.get(
        visitor_id,
        set()
    )


    history_list = list(history)


    item_indices = []
    item_values = []


    for item_id in history_list:

        if item_id in item_to_index:

            item_indices.append(
                item_to_index[item_id]
            )

            item_values.append(1.0)


    if len(item_indices) == 0:

        return (
            list(
                popular_items[
                    :n_recommendations
                ]
            ),
            "Popularity Fallback"
        )


    row_indices = np.zeros(
        len(item_indices),
        dtype=int
    )


    user_vector = csr_matrix(
        (
            item_values,
            (
                row_indices,
                item_indices
            )
        ),
        shape=(
            1,
            len(item_to_index)
        )
    )


    user_factors = svd_model.transform(
        user_vector
    )


    scores = np.dot(
        user_factors,
        svd_model.components_
    ).flatten()


    ranked_indices = np.argsort(
        scores
    )[::-1]


    seen_items = set(
        history_list
    )


    recommendations = []


    for item_index in ranked_indices:

        item_id = index_to_item[
            item_index
        ]


        if item_id not in seen_items:

            recommendations.append(
                item_id
            )


        if len(recommendations) >= n_recommendations:
            break


    if len(recommendations) < n_recommendations:

        for item_id in popular_items:

            if (
                item_id not in seen_items
                and item_id not in recommendations
            ):

                recommendations.append(
                    item_id
                )


            if len(recommendations) >= n_recommendations:
                break


    return (
        recommendations,
        "Personalized SVD"
    )


# Sidebar

st.sidebar.markdown(
    "## 🛍️ Recommendation Settings"
)


st.sidebar.markdown(
    "Choose how many products you want to see."
)


n_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=10,
    value=10
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    **Recommendation Engine**

    • Collaborative Filtering

    • Truncated SVD

    • Popularity Fallback

    • User Segmentation
    """
)


st.sidebar.markdown("---")


st.sidebar.caption(
    "Machine Learning Recommendation System"
)


# Main header

st.markdown(
    """
    <div class="main-title">
        Personalized Product Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        Discover products based on user interaction
        patterns and personalized recommendation models.
    </div>
    """,
    unsafe_allow_html=True
)


# User input

st.markdown(
    """
    <div class="section-title">
        👤 Enter User Information
    </div>
    """,
    unsafe_allow_html=True
)


visitor_id_input = st.text_input(
    "Visitor ID",
    placeholder="Example: 1150086"
)


if visitor_id_input:

    try:

        visitor_id = int(
            visitor_id_input
        )


        recommendations, method = (
            get_recommendations(
                visitor_id,
                n_recommendations
            )
        )


        # User profile

        st.markdown(
            """
            <div class="section-title">
                📊 User Profile
            </div>
            """,
            unsafe_allow_html=True
        )


        if visitor_id in user_to_index:

            history = user_history.get(
                visitor_id,
                set()
            )


            history_list = list(
                history
            )


            user_row = user_segments[
                user_segments["visitorid"]
                == visitor_id
            ]


            if not user_row.empty:

                segment = str(
                    user_row.iloc[0][
                        "User Segment"
                    ]
                )

            else:

                segment = "Unknown"


            col1, col2, col3 = st.columns(3)


            with col1:

                st.markdown(
                    f"""<div class="info-card"><div class="info-label">Visitor ID</div><div class="info-value">{visitor_id}</div></div>""",
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    f"""<div class="info-card"><div class="info-label">Interactions</div><div class="info-value">{len(history_list)}</div></div>""",
                    unsafe_allow_html=True
                )


            with col3:

                st.markdown(
                    f"""<div class="info-card"><div class="info-label">User Segment</div><div class="info-value">{segment}</div></div>""",
                    unsafe_allow_html=True
                )


            # Interaction history

            with st.expander(
                "🔎 Previously Interacted Items"
            ):

                history_df = pd.DataFrame(
                    {
                        "Item ID":
                        history_list[:50]
                    }
                )


                st.dataframe(
                    history_df,
                    width="stretch",
                    hide_index=True
                )

        else:

            st.info(
                "This visitor is not available in the trained user set. Popular products are being used as a fallback."
            )


        # Personalized recommendations

        st.markdown(
            """
            <div class="section-title">
                ✨ Personalized Recommendations
            </div>
            """,
            unsafe_allow_html=True
        )


        method_html = f"""
<div class="method-card">
    Recommendation Method:
    <span class="method-name">{method}</span>
</div>
"""


        st.markdown(
            method_html,
            unsafe_allow_html=True
        )


        # Recommendation cards

        recommendation_columns = st.columns(2)


        for i, item_id in enumerate(
            recommendations
        ):

            card_html = f"""<div class="recommend-card"><div class="recommend-rank">Recommendation #{i + 1}</div><div class="recommend-item">🛒 Product ID: {item_id}</div></div>"""


            with recommendation_columns[
                i % 2
            ]:

                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )


        # Recommendation table

        with st.expander(
            "📋 View Recommendation Table"
        ):

            recommendation_df = pd.DataFrame(
                {
                    "Rank": range(
                        1,
                        len(
                            recommendations
                        ) + 1
                    ),

                    "Recommended Item":
                        recommendations
                }
            )


            st.dataframe(
                recommendation_df,
                width="stretch",
                hide_index=True
            )


        # Popular baseline

        st.markdown(
            """
            <div class="section-title">
                🔥 Popular Product Baseline
            </div>
            """,
            unsafe_allow_html=True
        )


        popular_df = pd.DataFrame(
            {
                "Rank": range(
                    1,
                    min(
                        n_recommendations,
                        len(popular_items)
                    ) + 1
                ),

                "Popular Item":
                    popular_items[
                        :n_recommendations
                    ]
            }
        )


        st.dataframe(
            popular_df,
            width="stretch",
            hide_index=True
        )


    except ValueError:

        st.error(
            "Please enter a valid numeric Visitor ID."
        )


    except Exception as error:

        st.error(
            f"Recommendation system error: {error}"
        )


# Model performance

st.markdown(
    """
    <div class="section-title">
        📈 Recommendation Model Performance
    </div>
    """,
    unsafe_allow_html=True
)


st.dataframe(
    model_comparison,
    width="stretch",
    hide_index=True
)


# User segment distribution

st.markdown(
    """
    <div class="section-title">
        👥 User Segment Distribution
    </div>
    """,
    unsafe_allow_html=True
)


segment_counts = (
    user_segments[
        "User Segment"
    ]
    .value_counts()
)


segment_columns = st.columns(
    len(segment_counts)
)


for column, (segment_name, count) in zip(
    segment_columns,
    segment_counts.items()
):

    with column:

        st.markdown(
            f"""<div class="segment-card"><div class="segment-name">{segment_name}</div><div class="segment-value">{count:,}</div></div>""",
            unsafe_allow_html=True
        )


# Simple segment chart

st.markdown("")


chart_data = pd.DataFrame(
    {
        "Users": segment_counts
    }
)


st.bar_chart(
    chart_data,
    width="stretch"
)


# Methodology

with st.expander(
    "📚 About the Recommendation System"
):

    st.markdown(
        """
        ### Recommendation Methodology

        The system uses e-commerce interaction data
        to generate personalized product recommendations.

        **Interaction weights**

        - View = 1
        - Add to cart = 3
        - Transaction = 5

        **Recommendation approaches evaluated**

        - Popularity Baseline
        - Collaborative Filtering
        - Content Based
        - Hybrid Recommendation

        **Collaborative filtering**

        Truncated SVD was applied to the user-item
        interaction matrix to learn latent user and
        product representations.

        **Popularity baseline**

        Popular products are used as a baseline and
        fallback method for users who cannot receive
        personalized recommendations.

        **Evaluation**

        The recommendation models were evaluated using:

        - Precision@10
        - Recall@10
        - NDCG@10

        A time-based train-test split was used during
        model evaluation.
        """
    )


# Download results

st.markdown(
    """
    <div class="section-title">
        📥 Download Results
    </div>
    """,
    unsafe_allow_html=True
)


with open(
    RESULTS_DIR / "recommendation_model_comparison.csv",
    "rb"
) as file:

    st.download_button(
        label="⬇️ Download Model Comparison",
        data=file,
        file_name="recommendation_model_comparison.csv",
        mime="text/csv",
        width="content"
    )


# Footer

st.markdown(
    """
    <div class="footer">
        Personalized Product Recommendation System
        <br>
        Machine Learning Assignment
    </div>
    """,
    unsafe_allow_html=True
)