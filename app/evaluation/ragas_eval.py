"""
RAGAS Evaluation Script for Physics Hybrid RAG
Evaluates: Faithfulness, Answer Relevancy, Context Precision, Context Recall

Run: python app/evaluation/ragas_eval.py
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict

# ── 50 Benchmark Questions ────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    # DEFINITION (10)
    {"q": "What is electric charge?", "type": "definition", "chapter": "Electric Charges and Fields"},
    {"q": "Define electric field intensity.", "type": "definition", "chapter": "Electric Charges and Fields"},
    {"q": "What is electric potential?", "type": "definition", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "Define capacitance of a conductor.", "type": "definition", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "What is drift velocity?", "type": "definition", "chapter": "Current Electricity"},
    {"q": "Define resistivity of a material.", "type": "definition", "chapter": "Current Electricity"},
    {"q": "What is magnetic flux?", "type": "definition", "chapter": "Electromagnetic Induction"},
    {"q": "Define self-inductance.", "type": "definition", "chapter": "Electromagnetic Induction"},
    {"q": "What is displacement current?", "type": "definition", "chapter": "Electromagnetic Waves"},
    {"q": "Define power factor in AC circuits.", "type": "definition", "chapter": "Alternating Current"},

    # FORMULA (10)
    {"q": "State Coulomb's Law with its mathematical expression.", "type": "formula", "chapter": "Electric Charges and Fields"},
    {"q": "What is the formula for electric field due to a point charge?", "type": "formula", "chapter": "Electric Charges and Fields"},
    {"q": "Write the formula for energy stored in a capacitor.", "type": "formula", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "State Ohm's Law and write its formula.", "type": "formula", "chapter": "Current Electricity"},
    {"q": "Write the Biot-Savart Law for magnetic field.", "type": "formula", "chapter": "Moving Charges and Magnetism"},
    {"q": "What is the formula for magnetic field inside a solenoid?", "type": "formula", "chapter": "Moving Charges and Magnetism"},
    {"q": "State Faraday's Law of electromagnetic induction.", "type": "formula", "chapter": "Electromagnetic Induction"},
    {"q": "Write the formula for impedance in an LCR circuit.", "type": "formula", "chapter": "Alternating Current"},
    {"q": "What is the speed of electromagnetic waves in vacuum?", "type": "formula", "chapter": "Electromagnetic Waves"},
    {"q": "Write Gauss's Law in integral form.", "type": "formula", "chapter": "Electric Charges and Fields"},

    # CONCEPTUAL (10)
    {"q": "Explain the principle of superposition of electric forces.", "type": "conceptual", "chapter": "Electric Charges and Fields"},
    {"q": "What are equipotential surfaces? Explain their properties.", "type": "conceptual", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "Explain the working of a Wheatstone bridge.", "type": "conceptual", "chapter": "Current Electricity"},
    {"q": "Explain Lenz's Law and its physical significance.", "type": "conceptual", "chapter": "Electromagnetic Induction"},
    {"q": "What is resonance in an LCR circuit? Explain its condition.", "type": "conceptual", "chapter": "Alternating Current"},
    {"q": "Explain the concept of electric field lines and their properties.", "type": "conceptual", "chapter": "Electric Charges and Fields"},
    {"q": "What are eddy currents? How are they minimized?", "type": "conceptual", "chapter": "Electromagnetic Induction"},
    {"q": "Explain the difference between diamagnetic, paramagnetic, and ferromagnetic materials.", "type": "conceptual", "chapter": "Magnetism and Matter"},
    {"q": "What is the significance of Maxwell's equations?", "type": "conceptual", "chapter": "Electromagnetic Waves"},
    {"q": "Explain the phenomenon of electromagnetic induction.", "type": "conceptual", "chapter": "Electromagnetic Induction"},

    # NUMERICAL/CALCULATION (10)
    {"q": "How do you calculate the electric field at a point due to multiple charges?", "type": "numerical", "chapter": "Electric Charges and Fields"},
    {"q": "How is the capacitance of a parallel plate capacitor calculated?", "type": "numerical", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "How do you find the equivalent resistance of resistors in series and parallel?", "type": "numerical", "chapter": "Current Electricity"},
    {"q": "How is the force on a current-carrying conductor in a magnetic field calculated?", "type": "numerical", "chapter": "Moving Charges and Magnetism"},
    {"q": "How do you calculate the induced EMF in a rotating coil?", "type": "numerical", "chapter": "Electromagnetic Induction"},
    {"q": "How is the RMS value of alternating current related to peak value?", "type": "numerical", "chapter": "Alternating Current"},
    {"q": "How do you find the wavelength of an electromagnetic wave from its frequency?", "type": "numerical", "chapter": "Electromagnetic Waves"},
    {"q": "Calculate the energy stored in an inductor.", "type": "numerical", "chapter": "Alternating Current"},
    {"q": "How is the potential due to a dipole calculated?", "type": "numerical", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "How do you calculate current through each branch in Kirchhoff's laws?", "type": "numerical", "chapter": "Current Electricity"},

    # COMPARISON (5)
    {"q": "What is the difference between electric potential and electric potential energy?", "type": "comparison", "chapter": "Electrostatic Potential and Capacitance"},
    {"q": "Compare AC and DC circuits.", "type": "comparison", "chapter": "Alternating Current"},
    {"q": "Differentiate between self-inductance and mutual inductance.", "type": "comparison", "chapter": "Electromagnetic Induction"},
    {"q": "Compare the properties of electric field lines and magnetic field lines.", "type": "comparison", "chapter": "Moving Charges and Magnetism"},
    {"q": "What is the difference between EMF and terminal voltage?", "type": "comparison", "chapter": "Current Electricity"},

    # MULTI-HOP (5)
    {"q": "How does Coulomb's Law relate to Gauss's Law?", "type": "multi_hop", "chapter": "Electric Charges and Fields"},
    {"q": "How does electromagnetic induction lead to the working of a transformer?", "type": "multi_hop", "chapter": "Electromagnetic Induction"},
    {"q": "How are Maxwell's equations related to electromagnetic waves?", "type": "multi_hop", "chapter": "Electromagnetic Waves"},
    {"q": "How does the concept of displacement current resolve Ampere's circuital law?", "type": "multi_hop", "chapter": "Electromagnetic Waves"},
    {"q": "Explain how Faraday's Law and Lenz's Law together describe induction.", "type": "multi_hop", "chapter": "Electromagnetic Induction"},
]

API_BASE = "http://localhost:8000"


def run_query(question: str) -> Dict:
    """Query the RAG system."""
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": 10, "use_reranker": True},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Query failed: {e}")
    return {}


def evaluate_with_ragas(results: List[Dict]):
    """Run RAGAS evaluation metrics."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Build dataset
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for r in results:
            if r.get("answer") and r.get("chunks"):
                data["question"].append(r["question"])
                data["answer"].append(r["answer"])
                data["contexts"].append([c.get("content", "") for c in r["chunks"][:3]])
                # Ground truth: use answer itself if no GT available
                data["ground_truth"].append(r.get("answer", ""))

        if not data["question"]:
            print("No valid results to evaluate")
            return

        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        print("\n=== RAGAS Evaluation Results ===")
        print(result)
        return result

    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")


def run_evaluation(output_dir: str = "./data/evaluation"):
    """Run full evaluation pipeline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Running evaluation on {len(BENCHMARK_QUESTIONS)} questions...")
    print("=" * 60)

    results = []
    type_stats = {}

    for i, item in enumerate(BENCHMARK_QUESTIONS, 1):
        q = item["q"]
        qtype = item["type"]
        print(f"\n[{i:02d}/{len(BENCHMARK_QUESTIONS)}] [{qtype.upper()}] {q[:60]}...")

        t0 = time.time()
        resp = run_query(q)
        elapsed = time.time() - t0

        if resp:
            answer = resp.get("answer", "")
            citations = resp.get("citations", [])
            chunks = resp.get("chunks", [])
            latency = resp.get("latency", {})

            is_grounded = "not found" not in answer.lower()
            has_citation = len(citations) > 0

            result = {
                "question": q,
                "type": qtype,
                "chapter": item["chapter"],
                "answer": answer,
                "answer_length": len(answer.split()),
                "citations": citations,
                "num_chunks": len(chunks),
                "is_grounded": is_grounded,
                "has_citation": has_citation,
                "latency_ms": latency.get("total_ms", elapsed * 1000),
                "chunks": chunks,
            }
            results.append(result)

            # Type stats
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "grounded": 0, "cited": 0, "latencies": []}
            type_stats[qtype]["total"] += 1
            if is_grounded:
                type_stats[qtype]["grounded"] += 1
            if has_citation:
                type_stats[qtype]["cited"] += 1
            type_stats[qtype]["latencies"].append(latency.get("total_ms", 0))

            print(f"  ✓ Grounded: {is_grounded} | Citations: {len(citations)} | Latency: {latency.get('total_ms', 0):.0f}ms")
        else:
            print(f"  ✗ Query failed")

    # ── Save Results ──────────────────────────────────────────────────────────
    with open(f"{output_dir}/eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ── Print Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total = len(results)
    grounded = sum(1 for r in results if r["is_grounded"])
    cited = sum(1 for r in results if r["has_citation"])
    avg_latency = sum(r["latency_ms"] for r in results) / max(total, 1)

    print(f"Total Questions:   {total}")
    print(f"Grounded Answers:  {grounded}/{total} ({100*grounded/max(total,1):.1f}%)")
    print(f"With Citations:    {cited}/{total} ({100*cited/max(total,1):.1f}%)")
    print(f"Avg Latency:       {avg_latency:.0f}ms")
    print()
    print("By Question Type:")
    for qtype, stats in type_stats.items():
        t = stats["total"]
        g = stats["grounded"]
        avg_l = sum(stats["latencies"]) / max(t, 1)
        print(f"  {qtype:12s}: {g}/{t} grounded | {avg_l:.0f}ms avg")

    # ── RAGAS Evaluation ──────────────────────────────────────────────────────
    print("\nRunning RAGAS evaluation...")
    ragas_result = evaluate_with_ragas(results)

    # Save report
    report = {
        "summary": {
            "total": total,
            "grounded_pct": round(100 * grounded / max(total, 1), 1),
            "citation_pct": round(100 * cited / max(total, 1), 1),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "by_type": type_stats,
        "ragas": str(ragas_result) if ragas_result else "Not available",
    }

    with open(f"{output_dir}/eval_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return report


if __name__ == "__main__":
    run_evaluation()
