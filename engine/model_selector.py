from models.knowledge_base import KNOWLEDGE_BASE


CONCEPT_MAP = {
    "belonging"  : "overall_belonging_score",
    "stop out"   : "stop_out_flag",
    "stopout"    : "stop_out_flag",
    "graduation" : "graduated_flag",
    "graduate"   : "graduated_flag",
    "retention"  : "stop_out_flag",
    "gpa"        : "term_gpa_mean",
    "attendance" : "attendance_mean",
    "financial"  : "financial_hold_flag_max",
    "probation"  : "probation_flag_max",
    "income"     : "low_income_flag",
    "first gen"  : "first_gen_flag",
    "engagement" : "lms_login_mean"
}


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

    if any(k in q for k in predict_keywords):
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

    # match column names directly from manifest
    target_col = None
    for col in manifest["columns"]:
        col_hint = col.replace("_flag", "").replace("_score", "").replace("_", " ")
        if col_hint in q or col.replace("_", " ") in q:
            target_col = col
            break

    # concept keyword fallback
    if target_col is None:
        for concept, col in CONCEPT_MAP.items():
            if concept in q and col in manifest["columns"]:
                target_col = col
                break

    # final fallback for binary classification — default to stop_out_flag or first binary
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


def select_model(task_spec, manifest, summary):
    task_type  = task_spec["task_type"]
    target_col = task_spec["target_col"]
    n_rows     = summary["total_rows"]

    if task_type == "unknown":
        return {
            "status"  : "needs_clarification",
            "message" : "Could not understand the question. Please rephrase using words like predict, explain, correlate, group, or forecast."
        }

    if task_type not in KNOWLEDGE_BASE:
        return {
            "status"  : "unsupported",
            "message" : f"Task type '{task_type}' is not yet supported."
        }

    kb_entry = KNOWLEDGE_BASE[task_type]

    selected_model = None
    for model in kb_entry["models"]:
        if model["priority"] == 1 and n_rows > 300:
            selected_model = model
            break
        elif model["priority"] == 2:
            selected_model = model
            break

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

    return {
        "status"           : "ready",
        "task_type"        : task_type,
        "model"            : selected_model["name"],
        "reason"           : selected_model["why"],
        "target_col"       : target_col,
        "feature_cols"     : feature_cols,
        "expected_outputs" : selected_model["outputs"],
        "n_features"       : len(feature_cols),
        "n_rows"           : n_rows
    }


def run_selector(question, manifest, summary):
    print(f"\nQuestion : {question}")
    print("-" * 50)

    task_spec = parse_question(question, manifest, summary)
    print(f"Task type detected : {task_spec['task_type']}")
    print(f"Target column      : {task_spec['target_col']}")
    print(f"Confidence         : {task_spec['confidence']}")

    decision = select_model(task_spec, manifest, summary)

    if decision["status"] == "ready":
        print(f"\nModel selected  : {decision['model']}")
        print(f"Reason          : {decision['reason']}")
        print(f"Target column   : {decision['target_col']}")
        print(f"Feature columns : {decision['n_features']} columns")
        print(f"Expected output : {decision['expected_outputs']}")
    else:
        print(f"\nStatus  : {decision['status']}")
        print(f"Message : {decision['message']}")

    return decision
