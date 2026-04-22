# Flat model registry — each model is its own entry.
# Selection logic in engine/model_selector.py filters by task_type,
# checks hard_requirements, then scores by wins_when / loses_when.

MODEL_REGISTRY = {

    # ── Binary Classification ──────────────────────────────────────────────

    "xgboost_binary": {
        "name": "XGBoost",
        "task_types": ["classification_binary"],
        "hard_requirements": {"min_rows": 500, "min_features": 5},
        "outputs": ["accuracy", "roc_auc", "shap_values", "risk_scores"],
        "pros": [
            "Handles non-linearity and feature interactions",
            "Robust to mixed feature types (numeric + categorical)",
            "Built-in handling for imbalanced classes",
            "SHAP values give per-student explainability"
        ],
        "cons": [
            "Less interpretable than logistic regression",
            "Can overfit on very small datasets",
            "Harder to present to non-technical stakeholders"
        ],
        "wins_when": [
            "n_rows > 500",
            "n_features > 5",
            "has_outliers",
            "is_imbalanced"
        ],
        "loses_when": [
            "stakeholder_mode",
            "n_rows < 400"
        ]
    },

    "logistic_regression_binary": {
        "name": "Logistic Regression",
        "task_types": ["classification_binary"],
        "hard_requirements": {"min_rows": 50, "min_features": 2},
        "outputs": ["accuracy", "roc_auc", "odds_ratios", "coefficients"],
        "pros": [
            "Fully interpretable — outputs odds ratios",
            "Auditable by compliance and institutional review teams",
            "Works well on smaller datasets",
            "Fast to train and explain"
        ],
        "cons": [
            "Assumes linear decision boundary",
            "Struggles with feature interactions",
            "Performance drops on highly imbalanced targets"
        ],
        "wins_when": [
            "stakeholder_mode",
            "n_rows < 300"
        ],
        "loses_when": [
            "is_imbalanced",
            "n_features > 20",
            "has_outliers"
        ]
    },

    # ── Multi-class Classification ─────────────────────────────────────────

    "xgboost_multiclass": {
        "name": "XGBoost",
        "task_types": ["classification_multiclass"],
        "hard_requirements": {"min_rows": 500, "min_features": 5},
        "outputs": ["accuracy", "classification_report", "shap_values"],
        "pros": [
            "Handles multi-class natively via softmax",
            "Robust to mixed feature types",
            "Strong performance on tabular data"
        ],
        "cons": [
            "Less interpretable than Random Forest for small data",
            "Can overfit with few samples per class"
        ],
        "wins_when": [
            "n_rows > 300",
            "n_features > 5",
            "has_outliers"
        ],
        "loses_when": [
            "stakeholder_mode",
            "n_rows < 200"
        ]
    },

    "random_forest_multiclass": {
        "name": "Random Forest",
        "task_types": ["classification_multiclass"],
        "hard_requirements": {"min_rows": 100, "min_features": 3},
        "outputs": ["accuracy", "classification_report", "feature_importances"],
        "pros": [
            "Stable and less prone to overfitting on smaller datasets",
            "Feature importances are easy to communicate",
            "Handles outliers well via bootstrapping"
        ],
        "cons": [
            "Slower than XGBoost on large datasets",
            "Less precise on highly imbalanced classes"
        ],
        "wins_when": [
            "stakeholder_mode",
            "n_rows < 300",
            "has_outliers"
        ],
        "loses_when": [
            "is_imbalanced",
            "n_rows > 1000"
        ]
    },

    # ── Regression ─────────────────────────────────────────────────────────

    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "task_types": ["regression_continuous"],
        "hard_requirements": {"min_rows": 100, "min_features": 2},
        "outputs": ["rmse", "r2_score", "shap_values"],
        "pros": [
            "Best accuracy for numeric targets with complex relationships",
            "Handles outliers in features robustly",
            "SHAP values explain each prediction"
        ],
        "cons": [
            "Less interpretable than linear regression",
            "Requires more data to generalise well"
        ],
        "wins_when": [
            "n_rows > 300",
            "n_features > 5",
            "has_outliers"
        ],
        "loses_when": [
            "stakeholder_mode",
            "n_rows < 200"
        ]
    },

    "linear_regression": {
        "name": "Linear Regression",
        "task_types": ["regression_continuous"],
        "hard_requirements": {"min_rows": 30, "min_features": 2},
        "outputs": ["rmse", "r2_score", "coefficients"],
        "pros": [
            "Simple and fully interpretable",
            "Coefficients are easy to present to stakeholders",
            "Good baseline for any regression task"
        ],
        "cons": [
            "Assumes linear relationship between features and target",
            "Sensitive to outliers",
            "Struggles with many correlated features"
        ],
        "wins_when": [
            "stakeholder_mode",
            "n_rows < 300"
        ],
        "loses_when": [
            "has_outliers",
            "n_features > 20"
        ]
    },

    # ── Correlation Analysis ───────────────────────────────────────────────

    "correlation_matrix": {
        "name": "Correlation Matrix",
        "task_types": ["correlation_analysis"],
        "hard_requirements": {"min_rows": 30, "min_features": 2},
        "outputs": ["correlation_matrix", "top_correlations", "p_values"],
        "pros": [
            "Shows pairwise relationships across all numeric columns",
            "Fast and universally understood",
            "No training required"
        ],
        "cons": [
            "Only captures linear relationships",
            "Sensitive to outliers skewing correlation coefficients"
        ],
        "wins_when": [
            "n_features > 2"
        ],
        "loses_when": [
            "has_outliers"
        ]
    },

    # ── Clustering ─────────────────────────────────────────────────────────

    "kmeans": {
        "name": "KMeans",
        "task_types": ["clustering"],
        "hard_requirements": {"min_rows": 50, "min_features": 2},
        "outputs": ["cluster_labels", "cluster_sizes", "cluster_profiles"],
        "pros": [
            "Fast and scalable to large datasets",
            "Produces compact, well-separated clusters",
            "Easy to explain cluster centroids to stakeholders"
        ],
        "cons": [
            "Requires specifying number of clusters (k) in advance",
            "Assumes spherical clusters of similar size",
            "Sensitive to outliers pulling centroids"
        ],
        "wins_when": [
            "n_rows > 100",
            "n_rows < 5000"
        ],
        "loses_when": [
            "has_outliers"
        ]
    },

    "dbscan": {
        "name": "DBSCAN",
        "task_types": ["clustering"],
        "hard_requirements": {"min_rows": 50, "min_features": 2},
        "outputs": ["cluster_labels", "outlier_count", "cluster_profiles"],
        "pros": [
            "Finds clusters of any shape",
            "Automatically identifies and flags outliers",
            "No need to specify number of clusters"
        ],
        "cons": [
            "Slow on large datasets",
            "Sensitive to epsilon parameter choice",
            "Struggles with clusters of varying density"
        ],
        "wins_when": [
            "has_outliers",
            "n_rows < 5000"
        ],
        "loses_when": [
            "n_rows > 5000"
        ]
    },

    # ── Factor Analysis ────────────────────────────────────────────────────

    "factor_analysis": {
        "name": "Factor Analysis",
        "task_types": ["factor_analysis"],
        "hard_requirements": {"min_rows": 100, "min_features": 5},
        "outputs": ["factor_loadings", "explained_variance", "factor_names"],
        "pros": [
            "Purpose-built for survey construct discovery",
            "Separates shared variance from unique variance",
            "Produces named, interpretable factors"
        ],
        "cons": [
            "Requires ordinal or continuous input — not suitable for binary columns",
            "Sensitive to sample size relative to number of items",
            "Factor naming requires domain knowledge"
        ],
        "wins_when": [
            "n_features > 5",
            "n_rows > 100"
        ],
        "loses_when": [
            "n_rows < 100",
            "n_features < 5"
        ]
    },

    "pca": {
        "name": "PCA",
        "task_types": ["factor_analysis"],
        "hard_requirements": {"min_rows": 50, "min_features": 3},
        "outputs": ["components", "explained_variance", "scree_plot"],
        "pros": [
            "Reduces many columns to fewer components",
            "Good for visualisation and dimensionality reduction",
            "No distributional assumptions"
        ],
        "cons": [
            "Components are linear combinations — hard to name intuitively",
            "Maximises variance, not interpretability"
        ],
        "wins_when": [
            "n_features > 10",
            "stakeholder_mode"
        ],
        "loses_when": [
            "n_features < 5"
        ]
    },

    # ── Time-to-Event ──────────────────────────────────────────────────────

    "survival_analysis": {
        "name": "Survival Analysis (Cox PH)",
        "task_types": ["time_to_event"],
        "hard_requirements": {
            "min_rows": 100,
            "min_features": 2,
            "requires_time_column": True
        },
        "outputs": [
            "survival_curve", "hazard_ratios", "median_survival_time",
            "risk_by_term", "cox_coefficients"
        ],
        "pros": [
            "Models WHEN an event happens, not just IF it happens",
            "Handles censored students — those still enrolled with no outcome yet",
            "Standard method in student retention and persistence research",
            "Identifies at which term in the student journey risk peaks",
            "Produces individual survival curves per student"
        ],
        "cons": [
            "Requires a time variable — term number or enrollment duration",
            "More complex to interpret than logistic regression",
            "Less familiar to non-research stakeholders"
        ],
        "wins_when": [
            "is_longitudinal",
            "has_time_column",
            "n_rows > 100"
        ],
        "loses_when": [
            "not is_longitudinal",
            "not has_time_column",
            "stakeholder_mode"
        ]
    },

    # ── Causal Inference ───────────────────────────────────────────────────

    "propensity_score_matching": {
        "name": "Propensity Score Matching",
        "task_types": ["causal_inference"],
        "hard_requirements": {
            "min_rows": 200,
            "min_features": 3,
            "requires_treatment_column": True
        },
        "outputs": [
            "average_treatment_effect", "matched_sample_balance",
            "outcome_comparison", "standardized_mean_differences",
            "love_plot_data"
        ],
        "pros": [
            "Controls for selection bias in program evaluation",
            "Answers: did this intervention actually work?",
            "Produces treatment effect estimates, not just correlations",
            "Defensible methodology for policy and budget decisions",
            "Standard in program evaluation research"
        ],
        "cons": [
            "Requires a clear treatment and control group column",
            "Assumes no unmeasured confounders exist",
            "Reduces usable sample size after matching",
            "More complex pipeline than standard predictive models"
        ],
        "wins_when": [
            "n_rows > 200",
            "stakeholder_mode"
        ],
        "loses_when": [
            "n_rows < 200"
        ]
    }
}


TASK_REGISTRY = {
    "classification_binary": {
        "description": "Predict a yes/no outcome",
        "example_questions": [
            "which students are at risk of stopping out",
            "who is likely to graduate",
            "predict if a student will be placed on probation"
        ]
    },
    "classification_multiclass": {
        "description": "Predict which category something belongs to",
        "example_questions": [
            "classify students into risk tiers",
            "predict which support service a student will use"
        ]
    },
    "regression_continuous": {
        "description": "Predict a numeric value",
        "example_questions": [
            "predict a student's GPA next term",
            "forecast belonging score"
        ]
    },
    "correlation_analysis": {
        "description": "Find relationships between variables",
        "example_questions": [
            "is GPA correlated with attendance",
            "what is the relationship between belonging and stop out"
        ]
    },
    "clustering": {
        "description": "Group similar records together with no predefined target",
        "example_questions": [
            "segment students into groups",
            "find natural groups in the data"
        ]
    },
    "factor_analysis": {
        "description": "Understand the latent structure of survey or construct data",
        "example_questions": [
            "what constructs underlie these survey responses",
            "which survey items group together"
        ]
    },
    "time_to_event": {
        "description": "Analyse when an event happens and what predicts its timing",
        "example_questions": [
            "at what point in their enrollment are students most likely to drop out",
            "which students are likely to stop out before completing 30 credits",
            "show me the survival curve for first gen students vs non first gen",
            "which term has the highest dropout risk"
        ]
    },
    "causal_inference": {
        "description": "Evaluate whether an intervention or program caused an outcome",
        "example_questions": [
            "did the tutoring program improve retention",
            "did early alert intervention reduce stop out",
            "what was the impact of the financial aid restructure on graduation rates",
            "does living on campus cause higher belonging scores"
        ]
    }
}
