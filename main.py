import os
import sys

# allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from engine.profiler import profile_dataset, get_dataset_summary
from engine.model_selector import run_selector

DATA_PATH = os.path.join(os.path.dirname(__file__), "student_master (2).csv")

QUESTIONS = [
    "which students are at risk of stopping out?",
    "what factors drive student belonging?",
    "is GPA correlated with attendance?",
    "segment students into similar groups",
    "tell me something about the data",
    "what drives belonging among first gen students?",
    "which students are likely to graduate?",
    "forecast student GPA next term",
]


def main():
    print("=" * 60)
    print("Predictive Analytics Engine")
    print("=" * 60)

    print(f"\nStep 1 — Profiling dataset: {DATA_PATH}")
    manifest = profile_dataset(DATA_PATH)
    summary  = get_dataset_summary(manifest)
    print(f"  Rows    : {summary['total_rows']}")
    print(f"  Columns : {summary['total_columns']}")
    print(f"  Types   : {summary['column_type_breakdown']}")

    print("\nStep 2 — Running model selector")
    print("=" * 60)

    for question in QUESTIONS:
        run_selector(question, manifest, summary)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
