import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from engine.profiler import profile_dataset, get_dataset_summary
from engine.model_selector import run_selector

DATA_PATH = os.path.join(os.path.dirname(__file__), "student_master (2).csv")

QUESTIONS = [
    # Original 8
    "which students are at risk of stopping out?",
    "what factors drive student belonging?",
    "is GPA correlated with attendance?",
    "segment students into similar groups",
    "tell me something about the data",
    "what drives belonging among first gen students?",
    "which students are likely to graduate?",
    "forecast student GPA next term",
    # New 4
    "at what point in their enrollment are students most likely to stop out?",
    "did the early alert intervention program reduce stop out rates?",
    "which first gen students are at highest risk of stopping out before completing 30 credits?",
    "what was the impact of the tutoring program on retention?",
]


def print_final_summary(results, summary):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    ready    = [r for r in results if r["status"] == "ready"]
    not_ready = [r for r in results if r["status"] != "ready"]

    print(f"\nTotal tests run   : {len(results)}")
    print(f"  Completed       : {len(ready)}")
    print(f"  Not completed   : {len(not_ready)}")

    model_counts = Counter(r["model"] for r in ready)
    print("\nModels selected:")
    for model, count in model_counts.most_common():
        print(f"  {model:<35} × {count}")

    print(f"\nImbalanced targets detected : {summary['imbalanced_targets'] or 'none'}")
    print(f"Longitudinal data           : {summary['is_longitudinal']}")
    print(f"Time column present         : {summary['has_time_column']}")
    print(f"Likely IPEDS data           : {summary['likely_ipeds_data']}")
    print(f"Suppressed data (null>15%)  : {summary['has_suppressed_data']}")
    print(f"Small dataset (<300 rows)   : {summary['has_small_subgroups']}")

    all_notes = []
    all_flags = {}
    for r in ready:
        all_notes.extend(r.get("context_notes", []))
        all_flags.update(r.get("context_flags", {}))

    unique_notes = sorted(set(all_notes))
    if unique_notes:
        print("\nContext notes raised:")
        for note in unique_notes:
            print(f"  • {note}")

    if all_flags:
        print("\nContext flags raised:")
        for flag, val in all_flags.items():
            print(f"  • {flag}: {val}")

    print("\n" + "=" * 60)
    print("Done.")


def main():
    print("=" * 60)
    print("Predictive Analytics Engine")
    print("=" * 60)

    print(f"\nStep 1 — Profiling dataset")
    manifest = profile_dataset(DATA_PATH)
    summary  = get_dataset_summary(manifest)
    print(f"  Rows              : {summary['total_rows']}")
    print(f"  Columns           : {summary['total_columns']}")
    print(f"  Types             : {summary['column_type_breakdown']}")
    print(f"  Imbalanced targets: {summary['imbalanced_targets']}")
    print(f"  Longitudinal      : {summary['is_longitudinal']}")
    print(f"  IPEDS data        : {summary['likely_ipeds_data']}")

    print("\nStep 2 — Running model selector")
    print("=" * 60)

    results = []
    for question in QUESTIONS:
        decision = run_selector(question, manifest, summary)
        results.append(decision)

    print_final_summary(results, summary)


if __name__ == "__main__":
    main()
