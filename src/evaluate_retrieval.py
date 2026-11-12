
import json
import sys
from pathlib import Path
from time import perf_counter

# Ensure project root is available for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieve import retrieve


EVALUATION_FILE = BASE_DIR / "data" / "retrieval_eval.json"
RESULTS_FILE = BASE_DIR / "data" / "retrieval_eval_results.json"

TOP_K = 3


def load_evaluation_dataset():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    return dataset


def is_relevant(result, expected_sources, expected_pages):
    """
    A result is relevant when:
    - Its source matches an expected source.
    - Its page matches an expected page.

    Both conditions must hold.
    """
    source = result.get("source", "")
    page = result.get("page")

    return (
        source in expected_sources
        and page in expected_pages
    )


def calculate_metrics(results):
    total_questions = len(results)
    total_hits = 0
    total_precision = 0.0
    total_reciprocal_rank = 0.0

    out_of_scope_questions = 0
    out_of_scope_with_results = 0

    for item in results:
        retrieved = item["retrieved"]
        expected_sources = item["expected_sources"]
        expected_pages = item["expected_pages"]

        if item["category"] == "out_of_scope":
            out_of_scope_questions += 1

            if retrieved:
                out_of_scope_with_results += 1

            continue

        relevant_flags = [
            is_relevant(result, expected_sources, expected_pages)
            for result in retrieved
        ]

        hits = sum(relevant_flags)

        if hits > 0:
            total_hits += 1

        total_precision += hits / TOP_K

        for rank, relevant in enumerate(relevant_flags, start=1):
            if relevant:
                total_reciprocal_rank += 1 / rank
                break

    scored_questions = total_questions - out_of_scope_questions

    if scored_questions == 0:
        scored_questions = 1

    return {
        "total_questions": total_questions,
        "scored_questions": total_questions - out_of_scope_questions,
        "hit_rate_at_3": round(total_hits / scored_questions, 4),
        "precision_at_3": round(total_precision / scored_questions, 4),
        "mean_reciprocal_rank": round(
            total_reciprocal_rank / scored_questions, 4
        ),
        "out_of_scope_questions": out_of_scope_questions,
        "out_of_scope_with_retrieved_results": out_of_scope_with_results
    }


def evaluate():
    dataset = load_evaluation_dataset()

    results = []

    print("\n" + "=" * 65)
    print("ATLASQUERY - RETRIEVAL EVALUATION")
    print("=" * 65)

    print(f"Evaluation questions: {len(dataset)}")
    print(f"Top K: {TOP_K}")

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_pages = item.get("expected_pages", [])
        category = item.get("category", "unknown")

        print("\n" + "-" * 65)
        print(f"Question {index}/{len(dataset)}")
        print(f"Category: {category}")
        print(f"Question: {question}")

        start_time = perf_counter()

        retrieved, timings = retrieve(
            question,
            top_k=TOP_K,
            candidate_k=20,
            use_reranker=True,
            return_timings=True,
            use_hybrid=True
        )

        latency = perf_counter() - start_time

        retrieved_results = []

        for rank, chunk in enumerate(retrieved, start=1):
            relevant = is_relevant(
                chunk,
                expected_sources,
                expected_pages
            )

            retrieved_results.append({
                "rank": rank,
                "id": chunk.get("id"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
                "rerank_score": chunk.get("rerank_score"),
                "relevant": relevant,
                "text_preview": chunk.get("text", "")[:300]
            })

            print(
                f"\n  Rank {rank}: "
                f"{chunk.get('source')} | "
                f"Page {chunk.get('page')} | "
                f"Score {chunk.get('rerank_score')}"
            )

            print(
                "  Relevance:",
                "YES" if relevant else "NO"
            )

        if category == "out_of_scope":
            status = (
                "OUT-OF-SCOPE: NO RESULTS"
                if not retrieved
                else "OUT-OF-SCOPE: RESULTS RETURNED - REVIEW REQUIRED"
            )
        else:
            status = (
                "HIT"
                if any(result["relevant"] for result in retrieved_results)
                else "MISS"
            )

        print(f"\n  Evaluation: {status}")
        print(f"  Retrieval latency: {latency:.3f}s")

        results.append({
            "id": item.get("id"),
            "question": question,
            "category": category,
            "expected_sources": expected_sources,
            "expected_pages": expected_pages,
            "retrieved": retrieved_results,
            "evaluation": status,
            "latency_seconds": round(latency, 4),
            "timings": timings
        })

    metrics = calculate_metrics(results)

    output = {
        "project": "AtlasQuery",
        "evaluation_type": "Hybrid Retrieval + Reranking",
        "top_k": TOP_K,
        "metrics": metrics,
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("FINAL EVALUATION METRICS")
    print("=" * 65)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nDetailed results saved to:\n{RESULTS_FILE}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as error:
        print(f"\nEvaluation failed: {error}")
        raise
