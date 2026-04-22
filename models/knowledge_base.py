KNOWLEDGE_BASE = {

    "classification_binary": {
        "description": "Predict a yes/no outcome",
        "example_questions": [
            "which students are at risk of stopping out",
            "who is likely to graduate",
            "predict if a student will be placed on probation"
        ],
        "requires": {
            "target_type": "binary",
            "min_rows": 100,
            "min_features": 2
        },
        "models": [
            {
                "name": "XGBoost",
                "priority": 1,
                "why": "Handles interactions, non-linearity, mixed feature types. Best for large tabular data.",
                "use_when": "rows > 300 and n_features > 5",
                "outputs": ["accuracy", "roc_auc", "shap_values", "risk_scores"]
            },
            {
                "name": "Logistic Regression",
                "priority": 2,
                "why": "Interpretable, auditable, works well on smaller datasets.",
                "use_when": "rows <= 300 or stakeholder needs explainable model",
                "outputs": ["accuracy", "roc_auc", "odds_ratios", "coefficients"]
            }
        ]
    },

    "classification_multiclass": {
        "description": "Predict which category something belongs to",
        "example_questions": [
            "classify students into risk tiers",
            "predict which support service a student will use",
            "what type of intervention does this student need"
        ],
        "requires": {
            "target_type": "categorical_low",
            "min_rows": 200,
            "min_features": 3
        },
        "models": [
            {
                "name": "XGBoost",
                "priority": 1,
                "why": "Handles multi-class natively, robust to mixed features.",
                "use_when": "rows > 300",
                "outputs": ["accuracy", "classification_report", "shap_values"]
            },
            {
                "name": "Random Forest",
                "priority": 2,
                "why": "Stable, less prone to overfitting on smaller datasets.",
                "use_when": "rows <= 300",
                "outputs": ["accuracy", "classification_report", "feature_importances"]
            }
        ]
    },

    "regression_continuous": {
        "description": "Predict a numeric value",
        "example_questions": [
            "predict a student's GPA next term",
            "forecast belonging score",
            "estimate how many credits a student will complete"
        ],
        "requires": {
            "target_type": "continuous",
            "min_rows": 100,
            "min_features": 2
        },
        "models": [
            {
                "name": "Gradient Boosting Regressor",
                "priority": 1,
                "why": "Best accuracy for numeric targets with complex relationships.",
                "use_when": "rows > 300",
                "outputs": ["rmse", "r2_score", "shap_values"]
            },
            {
                "name": "Linear Regression",
                "priority": 2,
                "why": "Simple, interpretable, good baseline.",
                "use_when": "rows <= 300 or need interpretability",
                "outputs": ["rmse", "r2_score", "coefficients"]
            }
        ]
    },

    "correlation_analysis": {
        "description": "Find relationships between variables",
        "example_questions": [
            "is GPA correlated with attendance",
            "what is the relationship between belonging score and stop out",
            "which variables are related to each other"
        ],
        "requires": {
            "target_type": "any",
            "min_rows": 30,
            "min_features": 2
        },
        "models": [
            {
                "name": "Correlation Matrix",
                "priority": 1,
                "why": "Shows pairwise relationships across all numeric columns.",
                "use_when": "always",
                "outputs": ["correlation_matrix", "top_correlations", "p_values"]
            }
        ]
    },

    "clustering": {
        "description": "Group similar records together with no predefined target",
        "example_questions": [
            "segment students into groups",
            "find natural groups in the data",
            "which students are similar to each other"
        ],
        "requires": {
            "target_type": "none",
            "min_rows": 50,
            "min_features": 2
        },
        "models": [
            {
                "name": "KMeans",
                "priority": 1,
                "why": "Fast, simple, works well on tabular data.",
                "use_when": "always",
                "outputs": ["cluster_labels", "cluster_sizes", "cluster_profiles"]
            },
            {
                "name": "DBSCAN",
                "priority": 2,
                "why": "Finds clusters of irregular shape, handles outliers.",
                "use_when": "data has outliers or irregular structure",
                "outputs": ["cluster_labels", "outlier_count", "cluster_profiles"]
            }
        ]
    },

    "factor_analysis": {
        "description": "Understand structure of survey or construct data",
        "example_questions": [
            "what constructs underlie these survey responses",
            "which survey items group together",
            "reduce these survey columns into factors"
        ],
        "requires": {
            "target_type": "ordinal",
            "min_rows": 100,
            "min_features": 5
        },
        "models": [
            {
                "name": "Factor Analysis",
                "priority": 1,
                "why": "Purpose built for survey construct discovery.",
                "use_when": "columns are ordinal survey scores",
                "outputs": ["factor_loadings", "explained_variance", "factor_names"]
            },
            {
                "name": "PCA",
                "priority": 2,
                "why": "Dimensionality reduction, good for visualisation.",
                "use_when": "need to reduce many columns to fewer components",
                "outputs": ["components", "explained_variance", "scree_plot"]
            }
        ]
    }
}
