import pandas as pd
import numpy as np
import json


def get_variable_type(series):
    n_unique = series.nunique()
    n_rows = len(series)
    col_name = series.name.lower()

    id_keywords = ['pidm', '_id', 'id_', 'spriden', 'student_id', 'record_id']
    if any(kw in col_name for kw in id_keywords):
        return "id"

    if series.dtype in ['float64', 'int64'] and n_unique == n_rows:
        return "id"

    date_keywords = ['date', 'time', 'year', 'month', 'day']
    if any(k in col_name for k in date_keywords):
        return "date"

    if n_unique == 2:
        return "binary"

    if series.dtype in ['float64', 'int64'] and n_unique > 10:
        return "continuous"

    if series.dtype in ['float64', 'int64'] and n_unique <= 10:
        return "ordinal"

    if series.dtype == 'object' and n_unique <= 20:
        return "categorical_low"

    if series.dtype == 'object' and n_unique > 20:
        return "categorical_high"

    return "unknown"


def profile_column(series):
    variable_type = get_variable_type(series)

    profile = {
        "dtype": str(series.dtype),
        "variable_type": variable_type,
        "null_count": int(series.isnull().sum()),
        "null_pct": round(series.isnull().mean() * 100, 2),
        "unique_values": int(series.nunique())
    }

    if variable_type in ["id", "date"]:
        return profile

    if series.dtype in ['float64', 'int64']:
        profile["min"] = round(float(series.min()), 4)
        profile["max"] = round(float(series.max()), 4)
        profile["mean"] = round(float(series.mean()), 4)
        profile["std"] = round(float(series.std()), 4)

    if variable_type == "binary":
        profile["minority_pct"] = round(float(series.value_counts(normalize=True).min() * 100), 2)

    if series.dtype == 'object':
        profile["top_values"] = series.value_counts().head(5).index.tolist()

    return profile


def profile_dataset(filepath):
    df = pd.read_csv(filepath)

    manifest = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "columns": {}
    }

    for col in df.columns:
        manifest["columns"][col] = profile_column(df[col])

    return manifest


def get_dataset_summary(manifest):
    type_counts = {}
    for col, info in manifest["columns"].items():
        vtype = info["variable_type"]
        type_counts[vtype] = type_counts.get(vtype, 0) + 1

    id_columns = [
        col for col, info in manifest["columns"].items()
        if info["variable_type"] == "id"
    ]
    date_columns = [
        col for col, info in manifest["columns"].items()
        if info["variable_type"] == "date"
    ]
    binary_columns = [
        col for col, info in manifest["columns"].items()
        if info["variable_type"] == "binary"
    ]

    # longitudinal: a student ID column has fewer unique values than total rows,
    # meaning the same student appears in multiple rows
    is_longitudinal = any(
        manifest["columns"][col]["unique_values"] < manifest["total_rows"]
        for col in id_columns
    )

    # time-aware: at least one column name suggests a time or term dimension
    time_keywords = ["term", "year", "cohort", "semester", "quarter", "period", "date", "time"]
    has_time_column = any(
        any(kw in col.lower() for kw in time_keywords)
        for col in manifest["columns"]
    )

    # imbalanced binary targets: minority class below 20%
    imbalanced_targets = [
        col for col, info in manifest["columns"].items()
        if info["variable_type"] == "binary"
        and info.get("minority_pct", 50) < 20
    ]

    # small dataset: subgroup slices will produce unreliable cell sizes
    has_small_subgroups = manifest["total_rows"] < 300

    # IPEDS context: column names match common IPEDS reporting fields
    ipeds_keywords = [
        "unitid", "institution", "ipeds", "cohort", "fall_enrollment",
        "graduation_rate", "retention_rate", "pell", "fafsa"
    ]
    likely_ipeds_data = any(
        any(kw in col.lower() for kw in ipeds_keywords)
        for col in manifest["columns"]
    )

    # suppressed data: high null rates suggest IPEDS cell suppression
    has_suppressed_data = any(
        info["null_pct"] > 15
        for info in manifest["columns"].values()
    )

    summary = {
        "total_rows"            : manifest["total_rows"],
        "total_columns"         : manifest["total_columns"],
        "column_type_breakdown" : type_counts,
        "has_binary_target"     : len(binary_columns) > 0,
        "id_columns"            : id_columns,
        "date_columns"          : date_columns,
        "binary_columns"        : binary_columns,
        "is_longitudinal"       : is_longitudinal,
        "has_time_column"       : has_time_column,
        "imbalanced_targets"    : imbalanced_targets,
        "has_small_subgroups"   : has_small_subgroups,
        "likely_ipeds_data"     : likely_ipeds_data,
        "has_suppressed_data"   : has_suppressed_data,
    }

    return summary
