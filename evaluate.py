"""
evaluate.py — Evaluation Script
Run: python evaluate.py
Output: evaluation_results.csv, evaluation_summary.txt

Gives you concrete metrics to cite in interviews and SOPs:
  "I evaluated my RAG pipeline and measured X% answer relevance at Y ms latency"
"""

import os
import csv
import time
from dotenv import load_dotenv
from rag_pipeline import load_rag_chain, get_answer

load_dotenv()

# Replace with questions your specific corpus can answer
TEST_CASES = [
    {
        "question":          "What is the main topic covered in the document?",
        "expected_keywords": [],
        "notes":             "Broad question — tests general retrieval"
    },
    {
        "question":          "Summarise the key points from the introduction.",
        "expected_keywords": [],
        "notes":             "Tests multi-chunk aggregation"
    },
    {
        "question":          "What year was this document published?",
        "expected_keywords": [],
        "notes":             "Tests metadata extraction"
    },
    {
        "question":          "What is a topic not covered in these documents?",
        "expected_keywords": ["don't have enough information", "not in"],
        "notes":             "Out-of-scope — should trigger fallback"
    },
]


def keyword_hit(answer, keywords):
    if not keywords:
        return None
    return any(kw.lower() in answer.lower() for kw in keywords)


def run_evaluation():
    print("=" * 50)
    print("RAG Chatbot -- Evaluation")
    print("=" * 50)
    chain   = load_rag_chain()
    results = []
    scored  = hits = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\nQ{i}: {tc['question']}")
        start   = time.time()
        result  = get_answer(chain, tc["question"])
        elapsed = time.time() - start
        answer  = result["answer"]
        hit     = keyword_hit(answer, tc["expected_keywords"])
        if hit is not None:
            scored += 1
            if hit:
                hits += 1
        print(f"  Answer ({elapsed:.1f}s): {answer[:100]}...")
        print(f"  Sources: {len(result['sources'])}")
        if hit is not None:
            print(f"  Match: {'Yes' if hit else 'No'}")
        results.append({
            "question": tc["question"], "answer": answer[:500],
            "sources_count": len(result["sources"]),
            "latency_s": round(elapsed, 2), "keyword_hit": hit,
            "notes": tc["notes"]
        })

    csv_path = "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    accuracy    = (hits / scored * 100) if scored > 0 else 0
    avg_latency = sum(r["latency_s"] for r in results) / len(results)

    summary = f"""
RAG Evaluation Summary
======================
Total questions : {len(TEST_CASES)}
Scored (with GT): {scored}
Keyword hits    : {hits}
Accuracy        : {accuracy:.1f}%
Avg latency     : {avg_latency:.2f}s
Results file    : {csv_path}
"""
    print(summary)
    with open("evaluation_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    run_evaluation()
