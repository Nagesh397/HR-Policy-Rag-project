"""Run the ten evaluation questions and write a simple JSON report."""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.query import ask_llm, embed_query, load_config  # noqa: E402
from backend.app.utils.vectorstore import rerank_points  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

STOP_WORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "in", "is",
    "it", "may", "must", "of", "on", "should", "that", "the", "through", "to",
    "up", "with", "yes",
}


def normalize_source(source):
    """Match source names even when the indexed file uses PDF instead of Markdown."""
    return Path(source).stem.lower().replace(" ", "_")


def answer_terms(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    terms = set()
    for word in words:
        if word in STOP_WORDS or len(word) <= 2:
            continue
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        terms.add(word)
    return terms


def answer_matches(answer, expected_answer, expected_source):
    answer_lower = answer.lower()
    if expected_source == "none":
        refusal_terms = ("do not provide", "no information", "not provide", "does not provide")
        return any(term in answer_lower for term in refusal_terms)

    expected_terms = answer_terms(expected_answer)
    matched_terms = answer_terms(answer) & expected_terms
    return len(matched_terms) / max(len(expected_terms), 1) >= 0.6


def retrieve(client, config, question):
    query_vector = embed_query(question, config["ollama"]["embedding_model"])
    results = client.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=query_vector,
        limit=5,
    )
    return rerank_points(results.points, question, limit=3)


def format_context(points):
    context_parts = []
    sources = []
    for index, point in enumerate(points, start=1):
        payload = point.payload
        source = payload.get("source_file", "Unknown source")
        sources.append(source)
        page = payload.get("page_number")
        location = f"page {page}" if page else "document section"
        title = payload.get("document_title", source)
        context_parts.append(f"[{index}] {title}, {location}:\n{payload['text']}")
    return "\n\n".join(context_parts), sources


def evaluate_question(client, config, item):
    question = item["question"]
    expected_source = item["expected_source"]
    points = retrieve(client, config, question)
    context, sources = format_context(points)
    answer = ask_llm(context, question, config["ollama"]["llm_model"])

    retrieved_source_names = {normalize_source(source) for source in sources}
    expected_source_correct = (
        expected_source != "none"
        and normalize_source(expected_source) in retrieved_source_names
    )
    answer_quality = answer_matches(answer, item["expected_answer"], expected_source)

    return {
        "question": question,
        "generated_answer": answer.strip(),
        "expected_answer": item["expected_answer"],
        "retrieved_sources": sources,
        "retrieval_quality": "Good" if expected_source_correct else "Needs improvement",
        "answer_quality": "Good" if answer_quality else "Needs improvement",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("test_questions.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("evaluation_results.json"),
    )
    args = parser.parse_args()

    config = load_config()
    with args.questions.open(encoding="utf-8") as file:
        questions = yaml.safe_load(file)["questions"]

    client = QdrantClient(
        host=config["qdrant"]["host"],
        port=int(config["qdrant"]["port"]),
    )

    results = []
    for number, item in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {item['question']}")
        result = evaluate_question(client, config, item)
        results.append(result)
        print(f"  retrieval: {result['retrieval_quality']}")
        print(f"  answer: {result['answer_quality']}")

    retrieval_score = sum(item["retrieval_quality"] == "Good" for item in results)
    answer_score = sum(item["answer_quality"] == "Good" for item in results)

    report = {
        "question_count": len(results),
        "scores": {
            "retrieval_source_correct": {"earned": retrieval_score, "possible": len(results)},
            "answer_quality": {"earned": answer_score, "possible": len(results)},
        },
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"Retrieval quality: {retrieval_score}/{len(results)}")
    print(f"Answer quality: {answer_score}/{len(results)}")


if __name__ == "__main__":
    main()
