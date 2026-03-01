#!/usr/bin/env python3
"""BERTopic analysis for StackExchange Electronics questions.

This script:
1. Extracts questions from Posts.xml
2. Preprocesses text (HTML cleaning, combining Title + Body)
3. Applies BERTopic for topic clustering
4. Generates visualizations
5. Creates a user-friendly report for topic selection
"""

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from cuml import UMAP
from cuml.cluster import HDBSCAN

# Configuration
POSTS_XML_PATH = Path("data/raw/electronics.stackexchange.com/Posts.xml")
OUTPUT_DIR = Path("data/processed/electronics.stackexchange.com")
REPORT_PATH = OUTPUT_DIR / "topic_report"
VISUALIZATION_PATH = OUTPUT_DIR / "visualizations.html"
EMBEDDING_FILE = OUTPUT_DIR / "embeddings.json"

# BERTopic configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
VERBOSE = True

# Test configuration
MAX_QUESTIONS = None  # Set to a smaller number (e.g., 1000) for testing, or None for full dataset

# UMAP configuration
UMAP_N_NEIGHBORS = 15 if MAX_QUESTIONS is not None else 30
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# HDBSCAN configuration
HDBSCAN_MIN_CLUSTER_SIZE = 30 if MAX_QUESTIONS is not None else 150
HDBSCAN_MIN_SAMPLES = 15 if MAX_QUESTIONS is not None else 50
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# Vectorizer configuration
VECTORIZER_MIN_DF = 2 if MAX_QUESTIONS is not None else 5
VECTORIZER_MAX_DF = 0.90
VECTORIZER_NGRAM_RANGE = (1, 2)

# Outlier Threshold for reducing outliers
OUTLIER_THRESHOLD = 0.8

# Document preprocessing configuration
DOCUMENT_MAX_LENGTH = 5000

# Visualization configuration
VISUALIZATION_SAMPLE_SIZE = 5000

# Report configuration
REPORT_TITLE_TRUNCATION = 100
REPORT_TOP_KEYWORDS = 10
REPORT_TOP_WORDS = 3
REPORT_REPRESENTATIVE_QUESTIONS = 5


def extract_questions(
    xml_path: Path, max_questions: int | None = None
) -> list[dict]:
    """Extract questions from Posts.xml."""
    print(f"Extracting questions from {xml_path}...")

    questions = []
    context = ET.iterparse(str(xml_path), events=["start", "end"])
    context = iter(context)
    event, root = next(context)

    for event, elem in context:
        if event == "end" and elem.tag == "row":
            post_type = elem.get("PostTypeId")
            if post_type == "1":
                question = {
                    "id": elem.get("Id"),
                    "title": elem.get("Title", ""),
                    "body": elem.get("Body", ""),
                    "tags": elem.get("Tags", ""),
                    "score": int(elem.get("Score", 0)),
                    "view_count": int(elem.get("ViewCount", 0)),
                    "answer_count": int(elem.get("AnswerCount", 0)),
                    "creation_date": elem.get("CreationDate", ""),
                }
                questions.append(question)

                if max_questions and len(questions) >= max_questions:
                    break

            elem.clear()

    root.clear()
    print(f"Extracted {len(questions)} questions")
    return questions


def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_questions(questions: list[dict]) -> list[str]:
    """Preprocess questions by combining title and body, cleaning HTML."""
    print("Preprocessing questions...")

    documents = []
    for q in questions:
        title = clean_html(q.get("title", ""))
        body = clean_html(q.get("body", ""))
        combined = f"{title}. {body}"

        if len(combined) > DOCUMENT_MAX_LENGTH:
            combined = combined[:DOCUMENT_MAX_LENGTH] + "..."

        documents.append(combined)

    print(f"Preprocessed {len(documents)} documents")
    return documents


def create_topic_model() -> BERTopic:
    """Create and configure BERTopic model."""
    print(f"Initializing BERTopic with {EMBEDDING_MODEL}...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    representation_model = [
        KeyBERTInspired(top_n_words=20),
        MaximalMarginalRelevance(diversity=0.5),
    ]
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=VERBOSE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=VECTORIZER_MIN_DF,
        max_df=VECTORIZER_MAX_DF,
        ngram_range=VECTORIZER_NGRAM_RANGE,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics="auto",
        verbose=VERBOSE,
    )

    return topic_model


def generate_embeddings(
    topic_model: BERTopic, documents: list[str]
) -> np.ndarray:
    """Generate embeddings for documents."""
    print("Generating embeddings...")
    embeddings = topic_model.embedding_model.encode(
        documents, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings


def save_embeddings(question_ids: list[str], embeddings: np.ndarray):
    """Save embeddings to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_data = {
        "question_ids": question_ids,
        "embeddings": embeddings.tolist(),
    }
    with open(EMBEDDING_FILE, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f)
    print(f"Embeddings saved to {EMBEDDING_FILE}")


def load_embeddings() -> tuple[list[str], np.ndarray] | None:
    """Load embeddings from JSON file."""
    if not EMBEDDING_FILE.exists():
        return None

    print(f"Loading embeddings from {EMBEDDING_FILE}...")
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_ids = data["question_ids"]
    embeddings = np.array(data["embeddings"])
    print(
        f"Loaded {len(question_ids)} embeddings with shape {embeddings.shape}"
    )
    return question_ids, embeddings


def fit_topic_model(
    topic_model: BERTopic, documents: list[str], embeddings: np.ndarray
) -> tuple:
    """Fit BERTopic model to documents."""
    print(f"Fitting BERTopic on {len(documents)} documents...")
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)
    print(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
    len_outliers = sum(1 for t in topics if t == -1)
    if len_outliers > 0:
        print(f"{len_outliers} documents were classified as outliers")
        print("Reducing outliers...")
        new_topics = topic_model.reduce_outliers(
            documents,
            topics,
            strategy="embeddings",
            threshold=OUTLIER_THRESHOLD,
            embeddings=embeddings,
        )
        topics = new_topics
        topic_model.update_topics(
            documents,
            topics=topics,
            representation_model=topic_model.representation_model,
        )
        print(
            f"After reducing outliers, {sum(1 for t in topics if t == -1)} documents remain as outliers"
        )
    return topic_model, topics, probs


def generate_visualizations(
    topic_model: BERTopic, documents: list[str], topics: list[int]
):
    """Generate and save visualizations."""
    print("Generating visualizations...")

    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(str(OUTPUT_DIR / "inter_topic_distance.html"))

    if len(documents) > VISUALIZATION_SAMPLE_SIZE:
        np.random.seed(42)
        indices = np.random.choice(
            len(documents), VISUALIZATION_SAMPLE_SIZE, replace=False
        )
        docs_sample = [documents[i] for i in indices]
        topics_sample = [topics[i] for i in indices]
    else:
        docs_sample = documents
        topics_sample = topics

    try:
        fig_docs = topic_model.visualize_documents(
            docs_sample, topics=topics_sample
        )
        fig_docs.write_html(str(OUTPUT_DIR / "document_projections.html"))
    except Exception as e:
        print(f"Warning: Could not generate document projections: {e}")

    try:
        hierarchical_topics = topic_model.hierarchical_topics(docs_sample)
        fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics)
        fig_hierarchy.write_html(str(OUTPUT_DIR / "topic_hierarchy.html"))
    except Exception as e:
        print(f"Warning: Could not generate topic hierarchy: {e}")

    try:
        fig_barchart = topic_model.visualize_barchart()
        fig_barchart.write_html(str(OUTPUT_DIR / "topic_barchart.html"))
    except Exception as e:
        print(f"Warning: Could not generate topic barchart: {e}")

    print(f"Visualizations saved to {OUTPUT_DIR}")


def generate_report(
    topic_model: BERTopic,
    documents: list[str],
    topics: list[int],
    questions: list[dict],
):
    """Generate user-friendly report for topic selection."""
    print("Generating topic report...")

    topic_info = topic_model.get_topic_info()

    report_data = {
        "generated_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "total_topics": len(topic_info) - 1,
        "topics": [],
    }

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        count = row["Count"]

        if topic_id == -1:
            name = "Outliers"
            keywords = "N/A"
        else:
            topic = topic_model.get_topic(topic_id)
            keywords = " | ".join(
                [word for word, _ in topic[:REPORT_TOP_KEYWORDS]]
            )
            name = " | ".join([word for word, _ in topic[:REPORT_TOP_WORDS]])

        report_data["topics"].append(
            {
                "topic_id": int(topic_id),
                "name": name,
                "count": int(count),
                "keywords": keywords,
            }
        )

    with open(f"{REPORT_PATH}.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    save_topic_questions(topic_model, topics, questions, report_data)

    html_content = generate_html_report(
        topic_model, questions, topics, report_data
    )

    with open(f"{REPORT_PATH}.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to {REPORT_PATH}.json and {REPORT_PATH}.html")


def generate_html_report(
    topic_model: BERTopic,
    questions: list[dict],
    topics: list[int],
    report_data: dict,
) -> str:
    """Generate HTML report with topic details and representative questions."""
    topic_docs = {}
    for i, topic_id in enumerate(topics):
        if topic_id not in topic_docs:
            topic_docs[topic_id] = []
        topic_docs[topic_id].append(i)

    html_parts = []
    html_parts.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERTopic Analysis Report - Electronics Questions</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .topic-card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .topic-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
        .topic-id { background: #007bff; color: white; padding: 5px 12px; border-radius: 20px; }
        .topic-count { background: #28a745; color: white; padding: 5px 12px; border-radius: 20px; }
        .keywords { color: #666; font-style: italic; margin: 10px 0; }
        .question-item { padding: 10px; margin: 8px 0; background: #f8f9fa; border-left: 3px solid #007bff; }
        .question-title { font-weight: 600; color: #333; }
        .question-meta { font-size: 0.85em; color: #888; margin-top: 5px; }
        .filter-section { background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>BERTopic Analysis Report</h1>
    <p>Electronics StackExchange Questions Topic Analysis</p>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Documents:</strong> """
        + str(report_data["total_documents"])
        + """</p>
        <p><strong>Total Topics:</strong> """
        + str(report_data["total_topics"])
        + """</p>
        <p><strong>Generated:</strong> """
        + report_data["generated_at"]
        + """</p>
    </div>
    <div class="filter-section">
        <h3>Visualizations</h3>
        <ul>
            <li><a href="inter_topic_distance.html">Inter-topic Distance Map</a></li>
            <li><a href="document_projections.html">Document Projections (2D)</a></li>
            <li><a href="topic_hierarchy.html">Topic Hierarchy</a></li>
            <li><a href="topic_barchart.html">Topic Word Scores</a></li>
        </ul>
    </div>
    <h2>Topics</h2>
"""
    )

    sorted_topics = sorted(
        report_data["topics"],
        key=lambda x: x["count"] if x["topic_id"] != -1 else -1,
        reverse=True,
    )

    for topic_data in sorted_topics:
        topic_id = topic_data["topic_id"]
        name = topic_data["name"]
        count = topic_data["count"]
        keywords = topic_data["keywords"]

        html_parts.append(
            f"""
    <div class="topic-card">
        <div class="topic-header">
            <span class="topic-id">Topic {topic_id}</span>
            <span class="topic-count">{count} questions</span>
        </div>
        <h3>{name}</h3>
        <p class="keywords"><strong>Keywords:</strong> {keywords}</p>
"""
        )

        if topic_id in topic_docs and count > 0:
            doc_indices = topic_docs[topic_id]
            if len(doc_indices) > REPORT_REPRESENTATIVE_QUESTIONS:
                doc_indices = sorted(
                    doc_indices,
                    key=lambda i: questions[i].get("score", 0),
                    reverse=True,
                )[:REPORT_REPRESENTATIVE_QUESTIONS]

            html_parts.append("        <h4>Representative Questions:</h4>")
            for idx in doc_indices:
                q = questions[idx]
                title = q.get("title", "Untitled")[:REPORT_TITLE_TRUNCATION]
                score = q.get("score", 0)
                view_count = q.get("view_count", 0)
                answer_count = q.get("answer_count", 0)

                html_parts.append(
                    f"""
        <div class="question-item">
            <div class="question-title">{title}</div>
            <div class="question-meta">
                Score: {score} | Views: {view_count} | Answers: {answer_count}
            </div>
        </div>
"""
                )

        html_parts.append("    </div>")

    html_parts.append("</body></html>")

    return "".join(html_parts)


def save_topic_questions(
    topic_model: BERTopic,
    topics: list[int],
    questions: list[dict],
    report_data: dict,
):
    """Save each topic's questions to separate JSON files."""
    print("Saving topic questions to separate JSON files...")

    topic_docs = {}
    for i, topic_id in enumerate(topics):
        if topic_id not in topic_docs:
            topic_docs[topic_id] = []
        topic_docs[topic_id].append(i)

    topic_questions_dir = OUTPUT_DIR / "topic_questions"
    topic_questions_dir.mkdir(parents=True, exist_ok=True)

    topic_id_to_name = {t["topic_id"]: t["name"] for t in report_data["topics"]}

    for topic_data in report_data["topics"]:
        topic_id = topic_data["topic_id"]
        count = topic_data["count"]

        if topic_id == -1:
            continue

        if topic_id not in topic_docs or count == 0:
            continue

        doc_indices = topic_docs[topic_id]
        topic_questions = []

        for idx in doc_indices:
            q = questions[idx]
            topic_words = topic_model.get_topic(topic_id)
            keywords = (
                [word for word, _ in topic_words[:REPORT_TOP_KEYWORDS]]
                if topic_words
                else []
            )

            question_data = {
                "id": q.get("id"),
                "title": q.get("title"),
                "body": q.get("body"),
                "tags": q.get("tags"),
                "score": q.get("score"),
                "view_count": q.get("view_count"),
                "answer_count": q.get("answer_count"),
                "creation_date": q.get("creation_date"),
            }
            topic_questions.append(question_data)

        topic_questions.sort(key=lambda x: x.get("score", 0), reverse=True)

        topic_name = topic_id_to_name.get(topic_id, f"topic_{topic_id}")
        safe_name = re.sub(r"[^\w\-]", "_", topic_name)[:50]
        filename = f"topic_{topic_id:03d}_{safe_name}.json"
        filepath = topic_questions_dir / filename

        topic_json = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "keywords": topic_data["keywords"],
            "question_count": len(topic_questions),
            "questions": topic_questions,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(topic_json, f, indent=2, ensure_ascii=False)

    index_data = {
        "generated_at": report_data["generated_at"],
        "total_topics": report_data["total_topics"],
        "topics": [],
    }

    unified_topic_data = {
        "generated_at": report_data["generated_at"],
        "total_topics": report_data["total_topics"],
        "topics": [],
    }

    for topic_data in report_data["topics"]:
        topic_id = topic_data["topic_id"]
        if topic_id == -1:
            continue

        topic_name = topic_id_to_name.get(topic_id, f"topic_{topic_id}")
        safe_name = re.sub(r"[^\w\-]", "_", topic_name)[:50]
        filename = f"topic_{topic_id:03d}_{safe_name}.json"

        index_data["topics"].append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "keywords": topic_data["keywords"],
                "question_count": topic_data["count"],
                "filename": filename,
            }
        )

        if topic_id in topic_docs:
            question_ids = [
                questions[idx].get("id") for idx in topic_docs[topic_id]
            ]
        else:
            question_ids = []

        unified_topic_data["topics"].append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "keywords": topic_data["keywords"],
                "question_count": topic_data["count"],
                "question_ids": question_ids,
            }
        )

    index_data["topics"].sort(key=lambda x: x["question_count"], reverse=True)
    unified_topic_data["topics"].sort(
        key=lambda x: x["question_count"], reverse=True
    )

    index_path = topic_questions_dir / "topic_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    unified_path = OUTPUT_DIR / "topic_question_ids.json"
    with open(unified_path, "w", encoding="utf-8") as f:
        json.dump(unified_topic_data, f, indent=2, ensure_ascii=False)

    print(f"Saved topic questions to {topic_questions_dir}/")
    print(f"  - Topic index: topic_index.json")
    print(f"  - Individual topic files: topic_*.json")
    print(f"  - Unified question IDs: {unified_path}")


def save_model(topic_model: BERTopic):
    """Save BERTopic model to disk."""
    model_path = OUTPUT_DIR / "bertopic_model"
    topic_model.save(str(model_path))
    print(f"Model saved to {model_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BERTopic analysis for StackExchange Electronics questions"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation and load from existing files",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=MAX_QUESTIONS,
        help=f"Maximum number of questions to process (default: {MAX_QUESTIONS})",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract questions
    questions = extract_questions(
        POSTS_XML_PATH, max_questions=args.max_questions
    )
    question_ids = [q["id"] for q in questions]

    # Step 2: Preprocess
    documents = preprocess_questions(questions)

    # Step 3: Check for existing embeddings or generate new ones
    embeddings = None
    if args.skip_embeddings:
        loaded_data = load_embeddings()
        if loaded_data is not None:
            loaded_ids, embeddings = loaded_data
            if len(loaded_ids) != len(question_ids):
                print(
                    f"Warning: Loaded embeddings ({len(loaded_ids)}) don't match "
                    f"document count ({len(question_ids)}). Regenerating embeddings."
                )
                embeddings = None
            else:
                print("Successfully loaded existing embeddings!")

    topic_model = create_topic_model()

    if embeddings is not None:
        # Use loaded embeddings
        print(f"Using pre-computed embeddings with shape {embeddings.shape}")
        topic_model, topics, probs = fit_topic_model(
            topic_model, documents, embeddings
        )
    else:
        # Generate new embeddings
        embeddings = generate_embeddings(topic_model, documents)
        save_embeddings(question_ids, embeddings)
        topic_model, topics, probs = fit_topic_model(
            topic_model, documents, embeddings
        )

    # Step 4: Generate visualizations
    generate_visualizations(topic_model, documents, topics)

    # Step 5: Generate report
    generate_report(topic_model, documents, topics, questions)

    # Step 6: Save model
    save_model(topic_model)

    print("\nAnalysis complete!")
    print(f"   - Visualizations: {OUTPUT_DIR}/")
    print(f"   - Report: {REPORT_PATH}.html")
    print(f"   - Topic Questions: {OUTPUT_DIR}/topic_questions/")
    print(f"   - Embeddings: {EMBEDDING_FILE}")


if __name__ == "__main__":
    main()
