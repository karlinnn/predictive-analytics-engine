from models.knowledge_base import MODEL_REGISTRY, TASK_REGISTRY


CONCEPT_MAP = {
    "belonging"     : "overall_belonging_score",
    "connectedness" : "campus_connectedness",
    "faculty"       : "faculty_support",
    "peer"          : "peer_belonging",
    "institutional" : "institutional_support",
    "stop out"      : "stop_out_flag",
    "stopout"       : "stop_out_flag",
    "graduation"    : "graduated_flag",
    "graduate"      : "graduated_flag",
    "retention"     : "stop_out_flag",
    "gpa"           : "term_gpa_mean",
    "attendance"    : "attendance_mean",
    "financial"     : "financial_hold_flag_max",
    "probation"     : "probation_flag_max",
    "income"        : "low_income_flag",
    "first gen"     : "first_gen_flag",
    "engagement"    : "lms_login_mean",
    "tutoring"      : "tutoring_hours_sum",
    "advising"      : "intervention_flag_max",
    "intervention"  : "intervention_flag_max",
    "pell"          : "pell_eligible_flag",
    "fafsa"         : "fafsa_filed_flag",
    "transfer"      : "transfer_flag",
    "international" : "international_flag",
    "hispanic"      : "hispanic_flag",
    "cohort"        : "cohort_year",
    "term"          : "term_code",
    "credits"       : "credit_hours_cumulative_max",
    "scholarship"   : "scholarship_eligible_flag",
}


# ── Dataset characteristics ────────────────────────────────────────────────

def compute_characteristics(manifest, summary, target_col, n_features, stakeholder_mode=False):
    has_outliers = False
    for col, info in manifest["columns"].items():
        if info["variable_type"] == "continuous":
            if all(k in info for k in ["mean", "std", "max", "min"]):
                if info["max"] > info["mean"] + 3 * info["std"]:
                    has_outliers = True
                    break
                if info["min"] < info["mean"] - 3 * info["std"]:
                    has_outliers = True
                    break

    is_imbalanced = False
    if target_col and target_col in manifest["columns"]:
        col_info = manifest["columns"][target_col]
        if "minority_pct" in col_info:
            is_imbalanced = col_info["minority_pct"] < 20

    return {
        "n_rows"          : summary["total_rows"],
        "n_features"      : n_features,
        "has_outliers"    : has_outliers,
        "is_imbalanced"   : is_imbalanced,
        "is_longitudinal" : summary.get("is_longitudinal", False),
        "has_time_column" : summary.get("has_time_column", False),
        "stakeholder_mode": stakeholder_mode,
    }


# ── Condition evaluator ────────────────────────────────────────────────────

def evaluate_condition(condition, characteristics):
    condition = condition.strip()

    if condition.startswith("not "):
        return not evaluate_condition(condition[4:], characteristics)

    if condition in characteristics:
        return bool(characteristics[condition])

    for op in [" >= ", " <= ", " > ", " < ", " == "]:
        if op in condition:
            key, raw_value = condition.split(op, 1)
            char_value = characteristics.get(key.strip(), 0)
            threshold  = float(raw_value.strip())
            op = op.strip()
            if op == ">":  return char_value > threshold
            if op == "<":  return char_value < threshold
            if op == ">=": return char_value >= threshold
            if op == "<=": return char_value <= threshold
            if op == "==": return char_value == threshold

    return False


# ── Scoring ────────────────────────────────────────────────────────────────

def score_model(model_entry, characteristics):
    matched_wins   = [c for c in model_entry["wins_when"]  if evaluate_condition(c, characteristics)]
    matched_losses = [c for c in model_entry["loses_when"] if evaluate_condition(c, characteristics)]
    return len(matched_wins) - len(matched_losses), matched_wins, matched_losses


def compute_context_adjustment(model_id, summary, target_col, task_type):
    """University-specific score bonuses / penalties on top of wins_when / loses_when."""
    adj = 0

    if target_col and target_col in summary.get("imbalanced_targets", []):
        if "xgboost" in model_id:
            adj -= 2
        if "logistic_regression" in model_id:
            adj += 1

    if summary.get("is_longitudinal") and task_type == "time_to_event":
        if model_id == "survival_analysis":
            adj += 3

    if summary.get("has_small_subgroups"):
        if "xgboost" in model_id:
            adj -= 1
        if "logistic_regression" in model_id:
            adj += 1

    if summary.get("likely_ipeds_data"):
        if "logistic_regression" in model_id and task_type in [
            "classification_binary", "classification_multiclass"
        ]:
            adj += 1

    return adj


def compute_context_notes_flags(summary, target_col, task_type):
    notes = []
    flags = {}

    if target_col and target_col in summary.get("imbalanced_targets", []):
        flags["imbalanced_target"] = True

    if summary.get("is_longitudinal") and task_type == "time_to_event":
        notes.append("longitudinal data detected — survival analysis recommended")

    if summary.get("has_small_subgroups"):
        notes.append("small dataset — simpler model preferred")

    if summary.get("likely_ipeds_data"):
        notes.append("IPEDS data detected — auditability is important")

    if summary.get("has_suppressed_data"):
        notes.append("high null rates detected — check for IPEDS cell suppression before imputing")

    return notes, flags


# ── Question parser ────────────────────────────────────────────────────────

def parse_question(question, manifest, summary):
    q = question.lower().strip()

    predict_keywords   = ["predict", "risk", "likely", "who will", "which students",
                          "at risk", "stop out", "dropout", "flag", "will they"]
    explain_keywords   = ["why", "what drives", "what factors", "explain",
                          "most important", "top factors", "what causes"]
    correlate_keywords = ["correlated", "relationship", "related", "association",
                          "connection between", "linked"]
    cluster_keywords   = ["group", "segment", "cluster", "similar students",
                          "natural groups", "categorise"]
    forecast_keywords  = ["trend", "over time", "forecast", "future",
                          "next term", "predict gpa"]
    survey_keywords    = ["construct", "survey", "factor", "scale",
                          "underlying", "group items"]
    causal_keywords    = ["did the", "impact of", "effect of", "caused by",
                          "intervention", "program work", "does living",
                          "what was the impact", "evaluate the", "program improve"]
    time_event_keywords = ["when will", "which term", "survival", "how long until",
                           "at what point", "persistence trajectory", "term by term",
                           "before completing", "survival curve", "time to"]

    # order matters: more specific task types checked first
    if any(k in q for k in causal_keywords):
        task_type = "causal_inference"
    elif any(k in q for k in time_event_keywords):
        task_type = "time_to_event"
    elif any(k in q for k in predict_keywords):
        task_type = "classification_binary"
    elif any(k in q for k in explain_keywords):
        task_type = "classification_binary"
    elif any(k in q for k in correlate_keywords):
        task_type = "correlation_analysis"
    elif any(k in q for k in cluster_keywords):
        task_type = "clustering"
    elif any(k in q for k in forecast_keywords):
        task_type = "regression_continuous"
    elif any(k in q for k in survey_keywords):
        task_type = "factor_analysis"
    else:
        task_type = "unknown"

    target_col = None
    for col in manifest["columns"]:
        col_hint = col.replace("_flag", "").replace("_score", "").replace("_", " ")
        if col_hint in q or col.replace("_", " ") in q:
            target_col = col
            break

    if target_col is None:
        for concept, col in CONCEPT_MAP.items():
            if concept in q and col in manifest["columns"]:
                target_col = col
                break

    if target_col is None and task_type == "classification_binary":
        if "stop_out_flag" in summary["binary_columns"]:
            target_col = "stop_out_flag"
        elif summary["binary_columns"]:
            target_col = summary["binary_columns"][0]

    return {
        "task_type"  : task_type,
        "target_col" : target_col,
        "question"   : question,
        "confidence" : "high" if task_type != "unknown" else "low"
    }


# ── Model selector ─────────────────────────────────────────────────────────

def select_model(task_spec, manifest, summary, stakeholder_mode=False):
    task_type  = task_spec["task_type"]
    target_col = task_spec["target_col"]

    if task_type == "unknown":
        return {
            "status"  : "needs_clarification",
            "message" : "Could not understand the question. Please rephrase using words like "
                        "predict, explain, correlate, group, forecast, survival, or 'did the'."
        }

    if task_type not in TASK_REGISTRY:
        return {
            "status"  : "unsupported",
            "message" : f"Task type '{task_type}' is not yet supported."
        }

    exclude_cols = (
        summary["id_columns"] +
        summary["date_columns"] +
        ([target_col] if target_col else [])
    )
    feature_cols = [
        col for col, info in manifest["columns"].items()
        if col not in exclude_cols
        and info["variable_type"] not in ["id", "date"]
    ]
    n_features = len(feature_cols)

    characteristics = compute_characteristics(
        manifest, summary, target_col, n_features, stakeholder_mode
    )

    context_notes, context_flags = compute_context_notes_flags(summary, target_col, task_type)

    candidates = {
        model_id: entry
        for model_id, entry in MODEL_REGISTRY.items()
        if task_type in entry["task_types"]
    }

    eligible = {
        model_id: entry
        for model_id, entry in candidates.items()
        if characteristics["n_rows"]     >= entry["hard_requirements"]["min_rows"]
        and characteristics["n_features"] >= entry["hard_requirements"]["min_features"]
    }

    if not eligible:
        return {
            "status"  : "insufficient_data",
            "message" : (
                f"No model meets the minimum data requirements for '{task_type}'. "
                f"Need at least {min(e['hard_requirements']['min_rows'] for e in candidates.values())} rows "
                f"and {min(e['hard_requirements']['min_features'] for e in candidates.values())} features."
            )
        }

    scored = []
    for model_id, entry in eligible.items():
        base_score, wins, losses = score_model(entry, characteristics)
        context_adj = compute_context_adjustment(model_id, summary, target_col, task_type)
        final_score = base_score + context_adj
        scored.append((final_score, base_score, context_adj, model_id, entry, wins, losses))

    scored.sort(key=lambda x: x[0], reverse=True)
    final_score, base_score, context_adj, best_id, best_entry, best_wins, best_losses = scored[0]

    return {
        "status"           : "ready",
        "task_type"        : task_type,
        "model"            : best_entry["name"],
        "model_id"         : best_id,
        "score"            : final_score,
        "base_score"       : base_score,
        "context_adj"      : context_adj,
        "matched_wins"     : best_wins,
        "matched_losses"   : best_losses,
        "pros"             : best_entry["pros"],
        "cons"             : best_entry["cons"],
        "target_col"       : target_col,
        "feature_cols"     : feature_cols,
        "expected_outputs" : best_entry["outputs"],
        "n_features"       : n_features,
        "n_rows"           : characteristics["n_rows"],
        "characteristics"  : characteristics,
        "context_notes"    : context_notes,
        "context_flags"    : context_flags,
    }


# ── Pipeline entry point ───────────────────────────────────────────────────

def run_selector(question, manifest, summary, stakeholder_mode=False):
    print(f"\nQuestion : {question}")
    print("-" * 50)

    task_spec = parse_question(question, manifest, summary)
    print(f"Task type  : {task_spec['task_type']}")
    print(f"Target col : {task_spec['target_col']}")
    print(f"Confidence : {task_spec['confidence']}")

    decision = select_model(task_spec, manifest, summary, stakeholder_mode)

    if decision["status"] == "ready":
        adj_str = f"{decision['context_adj']:+d}" if decision["context_adj"] != 0 else "0"
        print(f"\nModel selected  : {decision['model']}")
        print(f"Score           : {decision['score']} "
              f"(base {decision['base_score']:+d}, context {adj_str})")
        print(f"Wins matched    : {decision['matched_wins']}")
        print(f"Losses matched  : {decision['matched_losses']}")
        print(f"Target column   : {decision['target_col']}")
        print(f"Feature columns : {decision['n_features']} columns")
        print(f"Expected output : {decision['expected_outputs']}")
        if decision["context_notes"]:
            for note in decision["context_notes"]:
                print(f"  [note] {note}")
        if decision["context_flags"]:
            for flag, val in decision["context_flags"].items():
                print(f"  [flag] {flag}: {val}")
    else:
        print(f"\nStatus  : {decision['status']}")
        print(f"Message : {decision['message']}")

    return decision
