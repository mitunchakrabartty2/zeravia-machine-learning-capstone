import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# load event data

events = pd.read_csv("events.csv")

print("First 5 rows:")
print(events.head())

print("\nDataset shape:")
print(events.shape)

print("\nColumn names:")
print(events.columns.tolist())

print("\nData types:")
print(events.dtypes)

print("\nMissing values:")
print(events.isnull().sum())

print("\nDuplicate rows:")
print(events.duplicated().sum())

print("\nEvent types:")
print(events["event"].value_counts())

print("\nNumber of unique visitors:")
print(events["visitorid"].nunique())

print("\nNumber of unique items:")
print(events["itemid"].nunique())


# clean the event data

events = events.drop_duplicates().copy()

events = events.dropna(
    subset=[
        "visitorid",
        "itemid",
        "event"
    ]
).copy()

events["visitorid"] = events[
    "visitorid"
].astype(np.int64)

events["itemid"] = events[
    "itemid"
].astype(np.int64)


# convert timestamp

events["datetime"] = pd.to_datetime(
    events["timestamp"],
    unit="ms"
)

events = events.sort_values(
    "datetime"
).reset_index(drop=True)

print("\nDate range:")
print(events["datetime"].min())
print(events["datetime"].max())


# assign interaction weights

event_weights = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}

events["interaction_weight"] = events[
    "event"
].map(event_weights)

events["interaction_weight"] = (
    events["interaction_weight"]
    .fillna(1.0)
)

print("\nWeighted interaction summary:")
print(
    events.groupby("event")[
        "interaction_weight"
    ].agg(["count", "mean"])
)


# interaction summary

interaction_summary = (
    events.groupby("event")
    .size()
    .reset_index(
        name="Interactions"
    )
)

print("\nInteraction summary:")
print(interaction_summary)


plt.figure(figsize=(8, 5))

plt.bar(
    interaction_summary["event"],
    interaction_summary["Interactions"]
)

plt.xlabel("Event Type")
plt.ylabel("Number of Interactions")
plt.title("User Interaction Distribution")

plt.tight_layout()
plt.show()


# time-based train and test split

split_time = events[
    "datetime"
].quantile(0.80)

train_events = events[
    events["datetime"] <= split_time
].copy()

test_events = events[
    events["datetime"] > split_time
].copy()

print("\nTime-based validation:")

print("Training period:")
print(
    train_events["datetime"].min(),
    "to",
    train_events["datetime"].max()
)

print("Testing period:")
print(
    test_events["datetime"].min(),
    "to",
    test_events["datetime"].max()
)

print("\nTraining interactions:")
print(len(train_events))

print("Testing interactions:")
print(len(test_events))


# keep active users and items

min_user_interactions = 10
min_item_interactions = 10

user_counts = (
    train_events
    .groupby("visitorid")
    .size()
)

item_counts = (
    train_events
    .groupby("itemid")
    .size()
)

active_users = user_counts[
    user_counts >= min_user_interactions
].index

active_items = item_counts[
    item_counts >= min_item_interactions
].index

train_events = train_events[
    train_events["visitorid"].isin(
        active_users
    )
].copy()

train_events = train_events[
    train_events["itemid"].isin(
        active_items
    )
].copy()

test_events = test_events[
    test_events["visitorid"].isin(
        active_users
    )
].copy()

test_events = test_events[
    test_events["itemid"].isin(
        active_items
    )
].copy()

print("\nAfter interaction filtering:")

print(
    "Training interactions:",
    len(train_events)
)

print(
    "Testing interactions:",
    len(test_events)
)

print(
    "Active users:",
    train_events[
        "visitorid"
    ].nunique()
)

print(
    "Active items:",
    train_events[
        "itemid"
    ].nunique()
)


# create mappings

user_ids = (
    train_events["visitorid"]
    .drop_duplicates()
    .tolist()
)

item_ids = (
    train_events["itemid"]
    .drop_duplicates()
    .tolist()
)

user_to_index = {
    user_id: index
    for index, user_id in enumerate(
        user_ids
    )
}

item_to_index = {
    item_id: index
    for index, item_id in enumerate(
        item_ids
    )
}

index_to_item = {
    index: item_id
    for item_id, index
    in item_to_index.items()
}


# aggregate user-item interactions

interaction_data = (
    train_events
    .groupby(
        [
            "visitorid",
            "itemid"
        ],
        as_index=False
    )["interaction_weight"]
    .sum()
)

row_indices = (
    interaction_data[
        "visitorid"
    ]
    .map(user_to_index)
    .values
)

column_indices = (
    interaction_data[
        "itemid"
    ]
    .map(item_to_index)
    .values
)

values = (
    interaction_data[
        "interaction_weight"
    ].values
)

user_item_matrix = csr_matrix(
    (
        values,
        (
            row_indices,
            column_indices
        )
    ),
    shape=(
        len(user_ids),
        len(item_ids)
    ),
    dtype=np.float32
)

print("\nSparse user-item matrix shape:")
print(user_item_matrix.shape)

print(
    "Non-zero interactions:",
    user_item_matrix.nnz
)


# matrix factorization

n_components = min(
    30,
    user_item_matrix.shape[0] - 1,
    user_item_matrix.shape[1] - 1
)

svd_model = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd_model.fit_transform(
    user_item_matrix
)

item_factors = svd_model.components_

print("\nMatrix factorization completed.")

print(
    "Number of components:",
    n_components
)

print(
    "Explained variance:",
    round(
        svd_model.explained_variance_ratio_.sum(),
        4
    )
)


# popularity baseline

popular_items = (
    train_events
    .groupby("itemid")
    .agg(
        Interaction_Count=(
            "itemid",
            "count"
        ),
        Weighted_Score=(
            "interaction_weight",
            "sum"
        )
    )
    .sort_values(
        "Weighted_Score",
        ascending=False
    )
)

popular_item_list = (
    popular_items.index.tolist()
)

print("\nTop popular items:")
print(
    popular_items.head(10)
)


# user history

user_history = (
    train_events
    .groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)


# collaborative filtering

def collaborative_recommend(
    user_id,
    top_k=10
):

    if user_id not in user_to_index:

        return []

    user_index = user_to_index[
        user_id
    ]

    scores = np.dot(
        user_factors[user_index],
        item_factors
    )

    scores = np.asarray(
        scores
    ).ravel()

    seen_items = user_history.get(
        user_id,
        set()
    )

    for item_id in seen_items:

        if item_id in item_to_index:

            scores[
                item_to_index[item_id]
            ] = -np.inf

    top_indices = np.argpartition(
        scores,
        -top_k
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(
            scores[top_indices]
        )[::-1]
    ]

    recommendations = []

    for index in top_indices:

        if np.isfinite(
            scores[index]
        ):

            recommendations.append(
                index_to_item[index]
            )

    return recommendations


# popularity recommendation

def popularity_recommend(
    user_id,
    top_k=10
):

    seen_items = user_history.get(
        user_id,
        set()
    )

    recommendations = []

    for item_id in popular_item_list:

        if item_id not in seen_items:

            recommendations.append(
                item_id
            )

        if len(recommendations) >= top_k:

            break

    return recommendations


# load item properties

content_item_ids_set = set(
    item_ids
)

item_content_parts = []

property_files = [
    "item_properties_part1.csv",
    "item_properties_part2.csv"
]

for file_name in property_files:

    print(
        "\nReading:",
        file_name
    )

    for chunk in pd.read_csv(
        file_name,
        usecols=[
            "itemid",
            "property",
            "value"
        ],
        chunksize=200000
    ):

        chunk = chunk[
            chunk["itemid"].isin(
                content_item_ids_set
            )
        ]

        if len(chunk) > 0:

            item_content_parts.append(
                chunk
            )


if len(item_content_parts) > 0:

    item_properties = pd.concat(
        item_content_parts,
        ignore_index=True
    )

else:

    item_properties = pd.DataFrame(
        columns=[
            "itemid",
            "property",
            "value"
        ]
    )


print("\nFiltered item properties:")
print(
    item_properties.shape
)


# prepare item content

if len(item_properties) > 0:

    item_properties["itemid"] = (
        item_properties["itemid"]
        .astype(int)
    )

    item_properties["property"] = (
        item_properties["property"]
        .astype(str)
    )

    item_properties["value"] = (
        item_properties["value"]
        .astype(str)
    )

    item_properties["content"] = (
        item_properties["property"]
        + "_"
        + item_properties["value"]
    )

    item_content = (
        item_properties
        .groupby("itemid")["content"]
        .apply(
            lambda x: " ".join(x)
        )
    )

else:

    item_content = pd.Series(
        dtype=str
    )


# TF-IDF content model

if len(item_content) > 10:

    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=2
    )

    item_content_matrix = (
        tfidf_vectorizer.fit_transform(
            item_content
        )
    )

    item_content_matrix = normalize(
        item_content_matrix
    )

    content_item_ids = (
        item_content.index.tolist()
    )

    content_item_to_index = {
        item_id: index
        for index, item_id
        in enumerate(
            content_item_ids
        )
    }

    print("\nContent matrix shape:")
    print(
        item_content_matrix.shape
    )

else:

    tfidf_vectorizer = None

    item_content_matrix = None

    content_item_ids = []

    content_item_to_index = {}


# content based recommendation

def content_recommend(
    user_id,
    top_k=10
):

    if item_content_matrix is None:

        return popularity_recommend(
            user_id,
            top_k
        )

    seen_items = user_history.get(
        user_id,
        set()
    )

    available_items = [
        item_id
        for item_id in seen_items
        if item_id in
        content_item_to_index
    ]

    if len(available_items) == 0:

        return popularity_recommend(
            user_id,
            top_k
        )

    profile_indices = [
        content_item_to_index[item_id]
        for item_id in available_items
    ]

    user_profile = (
        item_content_matrix[
            profile_indices
        ].mean(axis=0)
    )

    user_profile = csr_matrix(
        np.asarray(
            user_profile
        )
    )

    scores = (
        user_profile
        @ item_content_matrix.T
    )

    scores = np.asarray(
        scores.toarray()
    ).ravel()

    for item_id in seen_items:

        if item_id in content_item_to_index:

            scores[
                content_item_to_index[item_id]
            ] = -np.inf

    candidate_count = min(
        top_k,
        len(scores)
    )

    top_indices = np.argpartition(
        scores,
        -candidate_count
    )[-candidate_count:]

    top_indices = top_indices[
        np.argsort(
            scores[top_indices]
        )[::-1]
    ]

    recommendations = []

    for index in top_indices:

        if np.isfinite(
            scores[index]
        ):

            recommendations.append(
                content_item_ids[index]
            )

    return recommendations


# get collaborative scores

def get_collaborative_scores(
    user_id
):

    if user_id not in user_to_index:

        return None

    user_index = user_to_index[
        user_id
    ]

    scores = np.dot(
        user_factors[user_index],
        item_factors
    )

    return np.asarray(
        scores
    ).ravel()


# get content scores

def get_content_scores(
    user_id
):

    if item_content_matrix is None:

        return None

    seen_items = user_history.get(
        user_id,
        set()
    )

    available_items = [
        item_id
        for item_id in seen_items
        if item_id in
        content_item_to_index
    ]

    if len(available_items) == 0:

        return None

    profile_indices = [
        content_item_to_index[item_id]
        for item_id in available_items
    ]

    user_profile = (
        item_content_matrix[
            profile_indices
        ].mean(axis=0)
    )

    user_profile = csr_matrix(
        np.asarray(
            user_profile
        )
    )

    scores = (
        user_profile
        @ item_content_matrix.T
    )

    scores = np.asarray(
        scores.toarray()
    ).ravel()

    return scores


# score-based hybrid recommendation

def hybrid_recommend(
    user_id,
    top_k=10,
    collaborative_weight=0.6,
    content_weight=0.4
):

    collaborative_scores = (
        get_collaborative_scores(
            user_id
        )
    )

    content_scores = (
        get_content_scores(
            user_id
        )
    )

    if collaborative_scores is None:

        return popularity_recommend(
            user_id,
            top_k
        )

    if content_scores is None:

        return collaborative_recommend(
            user_id,
            top_k
        )

    content_values = np.zeros(
        len(item_ids),
        dtype=np.float32
    )

    for index, item_id in enumerate(
        content_item_ids
    ):

        if item_id in item_to_index:

            content_values[
                item_to_index[item_id]
            ] = content_scores[index]

    collaborative_min = (
        collaborative_scores.min()
    )

    collaborative_max = (
        collaborative_scores.max()
    )

    if (
        collaborative_max
        - collaborative_min
        > 0
    ):

        collaborative_normalized = (
            collaborative_scores
            - collaborative_min
        ) / (
            collaborative_max
            - collaborative_min
        )

    else:

        collaborative_normalized = (
            np.zeros_like(
                collaborative_scores
            )
        )

    content_min = (
        content_values.min()
    )

    content_max = (
        content_values.max()
    )

    if (
        content_max
        - content_min
        > 0
    ):

        content_normalized = (
            content_values
            - content_min
        ) / (
            content_max
            - content_min
        )

    else:

        content_normalized = (
            np.zeros_like(
                content_values
            )
        )

    hybrid_scores = (
        collaborative_weight
        * collaborative_normalized
        +
        content_weight
        * content_normalized
    )

    seen_items = user_history.get(
        user_id,
        set()
    )

    for item_id in seen_items:

        if item_id in item_to_index:

            hybrid_scores[
                item_to_index[item_id]
            ] = -np.inf

    top_indices = np.argpartition(
        hybrid_scores,
        -top_k
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(
            hybrid_scores[top_indices]
        )[::-1]
    ]

    recommendations = []

    for index in top_indices:

        if np.isfinite(
            hybrid_scores[index]
        ):

            recommendations.append(
                index_to_item[index]
            )

    return recommendations


# example recommendation

example_user = (
    train_events["visitorid"]
    .value_counts()
    .index[0]
)

print("\nExample user:")
print(example_user)

print("\nPopularity recommendations:")
print(
    popularity_recommend(
        example_user,
        10
    )
)

print("\nCollaborative recommendations:")
print(
    collaborative_recommend(
        example_user,
        10
    )
)

print("\nContent recommendations:")
print(
    content_recommend(
        example_user,
        10
    )
)

print("\nHybrid recommendations:")
print(
    hybrid_recommend(
        example_user,
        10
    )
)


# prepare test interactions

test_user_items = (
    test_events
    .groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)

evaluation_users = [
    user_id
    for user_id in test_user_items
    if user_id in user_to_index
]

max_evaluation_users = 200

if len(evaluation_users) > max_evaluation_users:

    evaluation_users = (
        evaluation_users[
            :max_evaluation_users
        ]
    )

print(
    "\nEvaluation users:",
    len(evaluation_users)
)


# evaluation metrics

def precision_at_k(
    recommended,
    actual,
    k
):

    recommended = recommended[:k]

    if len(recommended) == 0:

        return 0.0

    hits = len(
        set(recommended)
        & set(actual)
    )

    return hits / len(recommended)


def recall_at_k(
    recommended,
    actual,
    k
):

    if len(actual) == 0:

        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        & set(actual)
    )

    return hits / len(actual)


def ndcg_at_k(
    recommended,
    actual,
    k
):

    recommended = recommended[:k]

    if len(actual) == 0:

        return 0.0

    dcg = 0.0

    for rank, item_id in enumerate(
        recommended,
        start=1
    ):

        if item_id in actual:

            dcg += 1 / np.log2(
                rank + 1
            )

    ideal_hits = min(
        len(actual),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# evaluate recommendation methods

k = 10

evaluation_results = []

for counter, user_id in enumerate(
    evaluation_users,
    start=1
):

    actual_items = test_user_items[
        user_id
    ]

    popularity_items = (
        popularity_recommend(
            user_id,
            k
        )
    )

    collaborative_items = (
        collaborative_recommend(
            user_id,
            k
        )
    )

    content_items = (
        content_recommend(
            user_id,
            k
        )
    )

    hybrid_items = (
        hybrid_recommend(
            user_id,
            k
        )
    )

    evaluation_results.append({

        "UserID": user_id,

        "Popularity_Precision": (
            precision_at_k(
                popularity_items,
                actual_items,
                k
            )
        ),

        "Popularity_Recall": (
            recall_at_k(
                popularity_items,
                actual_items,
                k
            )
        ),

        "Popularity_NDCG": (
            ndcg_at_k(
                popularity_items,
                actual_items,
                k
            )
        ),

        "Collaborative_Precision": (
            precision_at_k(
                collaborative_items,
                actual_items,
                k
            )
        ),

        "Collaborative_Recall": (
            recall_at_k(
                collaborative_items,
                actual_items,
                k
            )
        ),

        "Collaborative_NDCG": (
            ndcg_at_k(
                collaborative_items,
                actual_items,
                k
            )
        ),

        "Content_Precision": (
            precision_at_k(
                content_items,
                actual_items,
                k
            )
        ),

        "Content_Recall": (
            recall_at_k(
                content_items,
                actual_items,
                k
            )
        ),

        "Content_NDCG": (
            ndcg_at_k(
                content_items,
                actual_items,
                k
            )
        ),

        "Hybrid_Precision": (
            precision_at_k(
                hybrid_items,
                actual_items,
                k
            )
        ),

        "Hybrid_Recall": (
            recall_at_k(
                hybrid_items,
                actual_items,
                k
            )
        ),

        "Hybrid_NDCG": (
            ndcg_at_k(
                hybrid_items,
                actual_items,
                k
            )
        )
    })

    if counter % 25 == 0:

        print(
            "Evaluated users:",
            counter,
            "/",
            len(evaluation_users)
        )


evaluation_table = pd.DataFrame(
    evaluation_results
)

print("\nRecommendation evaluation:")

print(
    evaluation_table.mean(
        numeric_only=True
    ).round(4)
)


# model comparison

model_comparison = pd.DataFrame({

    "Model": [
        "Popularity Baseline",
        "Collaborative Filtering",
        "Content Based",
        "Hybrid Recommendation"
    ],

    "Precision@10": [
        evaluation_table[
            "Popularity_Precision"
        ].mean(),

        evaluation_table[
            "Collaborative_Precision"
        ].mean(),

        evaluation_table[
            "Content_Precision"
        ].mean(),

        evaluation_table[
            "Hybrid_Precision"
        ].mean()
    ],

    "Recall@10": [
        evaluation_table[
            "Popularity_Recall"
        ].mean(),

        evaluation_table[
            "Collaborative_Recall"
        ].mean(),

        evaluation_table[
            "Content_Recall"
        ].mean(),

        evaluation_table[
            "Hybrid_Recall"
        ].mean()
    ],

    "NDCG@10": [
        evaluation_table[
            "Popularity_NDCG"
        ].mean(),

        evaluation_table[
            "Collaborative_NDCG"
        ].mean(),

        evaluation_table[
            "Content_NDCG"
        ].mean(),

        evaluation_table[
            "Hybrid_NDCG"
        ].mean()
    ]
})

print("\nModel comparison:")

print(
    model_comparison.round(4)
)


# user segmentation

user_statistics = (
    train_events
    .groupby("visitorid")
    .agg(
        Total_Interactions=(
            "itemid",
            "count"
        ),

        Unique_Items=(
            "itemid",
            "nunique"
        ),

        Transactions=(
            "event",
            lambda x: (
                x == "transaction"
            ).sum()
        ),

        Add_To_Cart=(
            "event",
            lambda x: (
                x == "addtocart"
            ).sum()
        )
    )
    .reset_index()
)


def assign_user_segment(row):

    if row["Transactions"] >= 3:

        return "Frequent Buyer"

    elif row["Transactions"] >= 1:

        return "Buyer"

    elif row["Total_Interactions"] >= 10:

        return "Active Browser"

    else:

        return "New or Casual User"


user_statistics[
    "User Segment"
] = user_statistics.apply(
    assign_user_segment,
    axis=1
)

print("\nUser segment distribution:")

print(
    user_statistics[
        "User Segment"
    ].value_counts()
)


# segment recommendations

segment_recommendations = []

for segment in user_statistics[
    "User Segment"
].unique():

    segment_users = (
        user_statistics[
            user_statistics[
                "User Segment"
            ] == segment
        ]["visitorid"]
        .tolist()
    )

    if len(segment_users) == 0:

        continue

    selected_user = segment_users[0]

    recommendations = hybrid_recommend(
        selected_user,
        10
    )

    for rank, item_id in enumerate(
        recommendations,
        start=1
    ):

        segment_recommendations.append({

            "User Segment": segment,

            "Example User": selected_user,

            "Rank": rank,

            "Recommended Item": item_id
        })


segment_recommendations = pd.DataFrame(
    segment_recommendations
)

print("\nSegment-based recommendations:")

print(
    segment_recommendations.head(20)
)


# recommendation report

recommendation_report = []

for user_id in evaluation_users[:100]:

    recommendations = hybrid_recommend(
        user_id,
        10
    )

    for rank, item_id in enumerate(
        recommendations,
        start=1
    ):

        recommendation_report.append({

            "UserID": user_id,

            "Rank": rank,

            "Recommended_Item": item_id
        })


recommendation_report = pd.DataFrame(
    recommendation_report
)


# save results

evaluation_table.to_csv(
    "recommendation_user_evaluation.csv",
    index=False
)

model_comparison.to_csv(
    "recommendation_model_comparison.csv",
    index=False
)

user_statistics.to_csv(
    "user_segments.csv",
    index=False
)

segment_recommendations.to_csv(
    "segment_recommendations.csv",
    index=False
)

recommendation_report.to_csv(
    "recommendation_report.csv",
    index=False
)

popular_items.reset_index().to_csv(
    "popular_items.csv",
    index=False
)


# save models

joblib.dump(
    svd_model,
    "recommendation_svd_model.pkl"
)

joblib.dump(
    tfidf_vectorizer,
    "recommendation_tfidf_vectorizer.pkl"
)

joblib.dump(
    user_to_index,
    "recommendation_user_mapping.pkl"
)

joblib.dump(
    item_to_index,
    "recommendation_item_mapping.pkl"
)

joblib.dump(
    index_to_item,
    "recommendation_index_to_item.pkl"
)

joblib.dump(
    popular_item_list,
    "recommendation_popular_items.pkl"
)

joblib.dump(
    user_history,
    "recommendation_user_history.pkl"
)


# save hybrid configuration

recommendation_config = {

    "top_k": 10,

    "svd_components":
        n_components,

    "minimum_user_interactions":
        min_user_interactions,

    "minimum_item_interactions":
        min_item_interactions,

    "collaborative_weight":
        0.6,

    "content_weight":
        0.4,

    "event_weights":
        event_weights,

    "evaluation_users":
        len(evaluation_users),

    "validation_method":
        "time-based 80-20 split"
}

joblib.dump(
    recommendation_config,
    "recommendation_config.pkl"
)


print("\nProject completed successfully.")

print("\nSaved files:")

print("recommendation_svd_model.pkl")
print("recommendation_tfidf_vectorizer.pkl")
print("recommendation_user_mapping.pkl")
print("recommendation_item_mapping.pkl")
print("recommendation_index_to_item.pkl")
print("recommendation_popular_items.pkl")
print("recommendation_user_history.pkl")
print("recommendation_config.pkl")
print("recommendation_model_comparison.csv")
print("recommendation_user_evaluation.csv")
print("recommendation_report.csv")
print("popular_items.csv")
print("user_segments.csv")
print("segment_recommendations.csv")