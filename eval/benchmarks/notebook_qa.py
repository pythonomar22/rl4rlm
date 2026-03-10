"""
Notebook-style QA benchmark for RLMs.

Simulates Jupyter notebook environments with interleaved code cells,
markdown explanations, and execution outputs. Tests whether the RLM
can navigate structured documents and answer questions about:
- Specific cell outputs
- Variable values computed across cells
- Patterns in execution results
- Connections between code logic and outputs

This is uniquely suited to RLMs: the model must parse code+output structure,
understand execution flow across cells, and sometimes chain information
from multiple cells.

O(K) complexity: must find and cross-reference K relevant cells in a notebook.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NotebookQATask:
    """A single Notebook QA task."""
    task_id: str
    prompt: str
    expected_answer: str
    n_cells: int
    question_type: str  # "output_lookup", "variable_trace", "pattern", "cross_cell"
    target_cells: list[int]  # Which cell indices contain relevant info
    doc_length: int


# Experiment names and contexts
EXPERIMENT_NAMES = [
    "customer_churn_analysis", "sales_forecast_q4", "sensor_anomaly_detection",
    "ab_test_results", "protein_folding_sim", "stock_portfolio_opt",
    "climate_data_trends", "user_engagement_metrics", "fraud_detection_v2",
    "nlp_sentiment_model", "supply_chain_opt", "energy_consumption_study",
]

AUTHOR_NAMES = [
    "Dr. Sarah Chen", "Prof. Michael Torres", "Dr. Aisha Patel",
    "James Morrison", "Dr. Lisa Nakamura", "Robert Blackwell",
    "Dr. Elena Vasquez", "Prof. David Kim", "Dr. Hannah Osei",
    "Marcus Johansson", "Dr. Priya Sharma", "Prof. Thomas Andersen",
]

DATASET_NAMES = [
    "customers_2024.csv", "transactions_q4.parquet", "sensor_readings.json",
    "experiment_log.csv", "protein_db.h5", "market_data.csv",
    "weather_stations.csv", "user_events.parquet", "fraud_labels.csv",
    "reviews_corpus.txt", "logistics_data.csv", "power_grid.csv",
]


def _generate_markdown_cell(rng: random.Random, topic: str) -> str:
    """Generate a realistic markdown cell."""
    templates = [
        f"## Data Preprocessing\n\nCleaning and preparing the {topic} dataset. "
        f"We handle missing values and normalize features.",
        f"## Exploratory Data Analysis\n\nLet's examine the distribution of key variables "
        f"in our {topic} data.",
        f"## Model Training\n\nFitting the model on our preprocessed {topic} dataset. "
        f"Using 80/20 train/test split with stratification.",
        f"## Results Analysis\n\nEvaluating model performance on the {topic} test set. "
        f"Key metrics below.",
        f"## Feature Importance\n\nAnalyzing which features contribute most to predictions "
        f"in our {topic} pipeline.",
        f"## Statistical Tests\n\nRunning significance tests on the {topic} results "
        f"to ensure findings are robust.",
        f"## Hyperparameter Tuning\n\nGrid search over key hyperparameters for optimal "
        f"{topic} performance.",
        f"## Conclusions\n\nSummary of findings from the {topic} analysis.",
    ]
    return rng.choice(templates)


def _generate_code_cell_filler(rng: random.Random) -> tuple[str, str]:
    """Generate a filler code cell with code and plausible output."""
    cells = [
        (
            "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\nprint('Libraries loaded successfully')",
            "Libraries loaded successfully",
        ),
        (
            "df.describe()",
            "       count     mean      std      min      25%      50%      75%      max\nage   10000.0   42.310   15.220    18.0    30.0    41.0    54.0    85.0\nsalary 10000.0  65421.3  28314.5  22000.0  44000.0  61000.0  82000.0  195000.0",
        ),
        (
            "print(f'Dataset shape: {df.shape}')\nprint(f'Columns: {list(df.columns)}')",
            "Dataset shape: (10000, 12)\nColumns: ['id', 'age', 'salary', 'region', 'tenure', 'status', 'score', 'segment', 'channel', 'visits', 'purchases', 'label']",
        ),
        (
            "# Check for missing values\ndf.isnull().sum()",
            "id          0\nage        23\nsalary     47\nregion      0\ntenure     12\nstatus      0\nscore      89\nsegment     0\nchannel     5\nvisits      0\npurchases   0\nlabel       0\ndtype: int64",
        ),
        (
            "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint(f'Scaled features shape: {X_scaled.shape}')",
            "Scaled features shape: (10000, 10)",
        ),
        (
            "# Correlation matrix\ncorr_matrix = df.select_dtypes(include=[np.number]).corr()\nprint('Top correlations with target:')\nprint(corr_matrix['label'].sort_values(ascending=False).head())",
            "Top correlations with target:\nlabel       1.000000\nscore       0.672341\npurchases   0.543218\nvisits      0.412987\ntenure      0.298456\nName: label, dtype: float64",
        ),
        (
            "plt.figure(figsize=(10, 6))\nplt.hist(df['age'], bins=30, edgecolor='black')\nplt.title('Age Distribution')\nplt.xlabel('Age')\nplt.ylabel('Count')\nplt.show()",
            "[Histogram plot displayed]",
        ),
        (
            "# Train/test split\nX_train, X_test, y_train, y_test = train_test_split(\n    X_scaled, y, test_size=0.2, random_state=42, stratify=y\n)\nprint(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')",
            "Train: 8000, Test: 2000",
        ),
        (
            "from sklearn.ensemble import GradientBoostingClassifier\nmodel = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)\nmodel.fit(X_train, y_train)\nprint('Model training complete')",
            "Model training complete",
        ),
        (
            "# Cross-validation\nfrom sklearn.model_selection import cross_val_score\ncv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')\nprint(f'CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')",
            "CV Accuracy: 0.8734 (+/- 0.0089)",
        ),
    ]
    return rng.choice(cells)


def _generate_output_lookup_task(
    task_idx: int,
    n_cells: int = 25,
    doc_length: int = 20000,
    seed: int = 42,
) -> NotebookQATask:
    """Generate a task where the answer is a specific output from a cell.

    The model must find a specific cell's output in a long notebook.
    """
    rng = random.Random(seed)

    experiment = rng.choice(EXPERIMENT_NAMES)
    author = rng.choice(AUTHOR_NAMES)
    dataset = rng.choice(DATASET_NAMES)

    # Generate target cell with distinctive output
    target_templates = [
        {
            "code": f"# Final model evaluation\ny_pred = model.predict(X_test)\nfrom sklearn.metrics import accuracy_score, f1_score\nacc = accuracy_score(y_test, y_pred)\nf1 = f1_score(y_test, y_pred, average='weighted')\nprint(f'Test Accuracy: {{acc:.4f}}')\nprint(f'Test F1 Score: {{f1:.4f}}')",
            "output_template": "Test Accuracy: {acc}\nTest F1 Score: {f1}",
            "question": f"What was the test F1 score reported in the {experiment} notebook?",
            "answer_key": "f1",
        },
        {
            "code": f"# Best hyperparameters from grid search\nprint('Best parameters found:')\nprint(f'  learning_rate: {{best_lr}}')\nprint(f'  max_depth: {{best_depth}}')\nprint(f'  n_estimators: {{best_n}}')\nprint(f'  Best CV score: {{best_score:.4f}}')",
            "output_template": "Best parameters found:\n  learning_rate: {lr}\n  max_depth: {depth}\n  n_estimators: {n_est}\n  Best CV score: {score}",
            "question": f"What learning rate was found to be best in the {experiment} hyperparameter search?",
            "answer_key": "lr",
        },
        {
            "code": f"# Feature importance\nimportances = model.feature_importances_\ntop_feature = feature_names[np.argmax(importances)]\nprint(f'Most important feature: {{top_feature}}')\nprint(f'Importance score: {{importances.max():.4f}}')",
            "output_template": "Most important feature: {feature}\nImportance score: {importance}",
            "question": f"What was the most important feature identified in the {experiment} analysis?",
            "answer_key": "feature",
        },
        {
            "code": f"# Final prediction count\nprint(f'Total predictions: {{len(y_pred)}}')\nprint(f'Positive predictions: {{(y_pred == 1).sum()}}')\nprint(f'Negative predictions: {{(y_pred == 0).sum()}}')\nprint(f'Positive rate: {{(y_pred == 1).mean():.2%}}')",
            "output_template": "Total predictions: {total}\nPositive predictions: {pos}\nNegative predictions: {neg}\nPositive rate: {rate}",
            "question": f"What was the positive prediction rate in the {experiment} results?",
            "answer_key": "rate",
        },
    ]

    template = rng.choice(target_templates)

    # Generate specific values
    values = {
        "acc": f"{rng.uniform(0.78, 0.95):.4f}",
        "f1": f"{rng.uniform(0.75, 0.93):.4f}",
        "lr": rng.choice(["0.01", "0.05", "0.001", "0.1", "0.005"]),
        "depth": str(rng.choice([3, 4, 5, 6, 7, 8])),
        "n_est": str(rng.choice([100, 150, 200, 300, 500])),
        "score": f"{rng.uniform(0.82, 0.94):.4f}",
        "feature": rng.choice(["purchase_frequency", "session_duration", "tenure_months",
                                "engagement_score", "total_spend", "recency_days"]),
        "importance": f"{rng.uniform(0.15, 0.35):.4f}",
        "total": str(rng.choice([1000, 1500, 2000, 2500])),
        "pos": str(rng.randint(300, 800)),
        "neg": str(rng.randint(700, 1700)),
        "rate": f"{rng.uniform(0.25, 0.45):.2%}",
    }

    target_output = template["output_template"].format(**values)
    answer = values[template["answer_key"]]

    # Place target cell at a random position
    target_pos = rng.randint(n_cells // 3, n_cells - 3)

    # Build notebook
    cells = []
    # Header
    cells.append(f"# {experiment.replace('_', ' ').title()}\n\n"
                 f"**Author:** {author}  \n**Dataset:** {dataset}  \n"
                 f"**Date:** 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")

    current_len = len(cells[0])
    cell_idx = 1

    while cell_idx < n_cells and current_len < doc_length:
        if cell_idx == target_pos:
            # Insert target cell
            cell_content = f"In [{cell_idx}]:\n{template['code']}\n\nOut [{cell_idx}]:\n{target_output}"
            cells.append(cell_content)
        elif rng.random() < 0.3:
            # Markdown cell
            md = _generate_markdown_cell(rng, experiment.replace('_', ' '))
            cells.append(md)
        else:
            # Code cell
            code, output = _generate_code_cell_filler(rng)
            cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")

        current_len += len(cells[-1])
        cell_idx += 1

    # Pad with filler if needed
    while current_len < doc_length:
        code, output = _generate_code_cell_filler(rng)
        cell_content = f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}"
        cells.append(cell_content)
        current_len += len(cell_content)
        cell_idx += 1

    notebook = "\n\n---\n\n".join(cells)
    prompt = f"QUESTION: {template['question']}\n\nNOTEBOOK:\n{notebook}"

    return NotebookQATask(
        task_id=f"notebook_output_{task_idx}",
        prompt=prompt,
        expected_answer=answer,
        n_cells=cell_idx,
        question_type="output_lookup",
        target_cells=[target_pos],
        doc_length=len(prompt),
    )


def _generate_variable_trace_task(
    task_idx: int,
    n_cells: int = 30,
    doc_length: int = 30000,
    seed: int = 42,
) -> NotebookQATask:
    """Generate a task requiring tracing a variable across multiple cells.

    The model must follow how a value is computed, modified, and used
    across several cells — a multi-hop reasoning task in code form.
    """
    rng = random.Random(seed)

    experiment = rng.choice(EXPERIMENT_NAMES)
    author = rng.choice(AUTHOR_NAMES)

    # Generate a computation chain
    chain_templates = [
        {
            "cells": [
                ("initial_samples = {n1}\nprint(f'Initial dataset: {initial_samples} samples')",
                 "Initial dataset: {n1} samples"),
                ("# Remove duplicates\nduplicates = {n2}\nclean_samples = initial_samples - duplicates\nprint(f'After dedup: {clean_samples} samples (removed {duplicates})')",
                 "After dedup: {clean} samples (removed {n2})"),
                ("# Apply filters\nfiltered_out = int(clean_samples * {frac})\nfinal_samples = clean_samples - filtered_out\nprint(f'Final dataset: {final_samples} samples')",
                 "Final dataset: {final} samples"),
            ],
            "question": f"How many samples remained in the final dataset after deduplication and filtering in the {experiment} notebook?",
            "compute_answer": lambda vals: str(vals["final"]),
        },
        {
            "cells": [
                ("base_accuracy = {base_acc}\nprint(f'Baseline accuracy: {base_acc:.1%}')",
                 "Baseline accuracy: {base_pct}"),
                ("# After feature engineering\nimproved_accuracy = base_accuracy + {delta1}\nprint(f'After feature eng: {improved_accuracy:.1%}')",
                 "After feature eng: {mid_pct}"),
                ("# After hyperparameter tuning\nfinal_accuracy = improved_accuracy + {delta2}\nprint(f'Final accuracy: {final_accuracy:.1%}')",
                 "Final accuracy: {final_pct}"),
            ],
            "question": f"What was the final accuracy after all improvements in the {experiment} notebook?",
            "compute_answer": lambda vals: vals["final_pct"],
        },
    ]

    template = rng.choice(chain_templates)

    # Generate values
    n1 = rng.choice([10000, 15000, 20000, 25000, 50000])
    n2 = rng.randint(200, 2000)
    clean = n1 - n2
    frac = rng.choice([0.1, 0.15, 0.2, 0.25])
    final = clean - int(clean * frac)

    base_acc = rng.uniform(0.65, 0.78)
    delta1 = rng.uniform(0.03, 0.08)
    delta2 = rng.uniform(0.02, 0.05)
    final_acc = base_acc + delta1 + delta2

    values = {
        "n1": n1, "n2": n2, "clean": clean, "frac": frac, "final": final,
        "base_acc": base_acc, "base_pct": f"{base_acc:.1%}",
        "delta1": delta1, "delta2": delta2,
        "mid_pct": f"{base_acc + delta1:.1%}",
        "final_pct": f"{final_acc:.1%}",
        "final_accuracy": final_acc, "improved_accuracy": base_acc + delta1,
        "initial_samples": n1, "clean_samples": clean,
        "filtered_out": int(clean * frac), "final_samples": final,
        "duplicates": n2,
    }

    answer = template["compute_answer"](values)

    # Place chain cells at specific positions with filler between
    chain_positions = sorted(rng.sample(range(3, n_cells - 2), len(template["cells"])))

    cells = []
    cells.append(f"# {experiment.replace('_', ' ').title()}\n\n**Author:** {author}")

    current_len = len(cells[0])
    chain_idx = 0
    cell_idx = 1

    while cell_idx < n_cells and current_len < doc_length:
        if chain_idx < len(chain_positions) and cell_idx == chain_positions[chain_idx]:
            code_tmpl, out_tmpl = template["cells"][chain_idx]
            code = code_tmpl.format(**values)
            output = out_tmpl.format(**values)
            cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")
            chain_idx += 1
        elif rng.random() < 0.25:
            md = _generate_markdown_cell(rng, experiment.replace('_', ' '))
            cells.append(md)
        else:
            code, output = _generate_code_cell_filler(rng)
            cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")

        current_len += len(cells[-1])
        cell_idx += 1

    while current_len < doc_length:
        code, output = _generate_code_cell_filler(rng)
        cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")
        current_len += len(cells[-1])
        cell_idx += 1

    notebook = "\n\n---\n\n".join(cells)
    prompt = f"QUESTION: {template['question']}\n\nNOTEBOOK:\n{notebook}"

    return NotebookQATask(
        task_id=f"notebook_trace_{task_idx}",
        prompt=prompt,
        expected_answer=answer,
        n_cells=cell_idx,
        question_type="variable_trace",
        target_cells=chain_positions,
        doc_length=len(prompt),
    )


def _generate_cross_cell_task(
    task_idx: int,
    n_cells: int = 35,
    doc_length: int = 40000,
    seed: int = 42,
) -> NotebookQATask:
    """Generate a task requiring cross-referencing information from multiple cells.

    The model must find results in one cell and relate them to
    definitions/explanations in another cell.
    """
    rng = random.Random(seed)

    experiment = rng.choice(EXPERIMENT_NAMES)
    author = rng.choice(AUTHOR_NAMES)

    # Generate models with results
    model_names = rng.sample([
        "RandomForest", "XGBoost", "LightGBM", "CatBoost",
        "LogisticRegression", "SVM", "NeuralNet", "KNN",
    ], 4)

    accuracies = sorted([rng.uniform(0.72, 0.95) for _ in range(4)], reverse=True)
    best_model = model_names[0]
    best_acc = accuracies[0]

    # Cell 1: Model definitions
    config_cell_code = "models = {\n"
    for name in model_names:
        config_cell_code += f"    '{name}': {name}Classifier(**best_params['{name}']),\n"
    config_cell_code += "}\nprint(f'Configured {len(models)} models for comparison')"
    config_cell_out = f"Configured {len(model_names)} models for comparison"

    # Cell 2: Results table
    results_code = "results = {}\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    acc = accuracy_score(y_test, model.predict(X_test))\n    results[name] = acc\n\nfor name, acc in sorted(results.items(), key=lambda x: -x[1]):\n    print(f'{name}: {acc:.4f}')"
    results_out = "\n".join(f"{model_names[i]}: {accuracies[i]:.4f}" for i in range(4))

    # Cell 3: Winner declaration
    winner_code = f"best_model = max(results, key=results.get)\nprint(f'Best model: {{best_model}} with accuracy {{results[best_model]:.4f}}')"
    winner_out = f"Best model: {best_model} with accuracy {best_acc:.4f}"

    question_templates = [
        (f"Which model achieved the highest accuracy in the {experiment} comparison, and what was its score?",
         f"{best_model} with accuracy {best_acc:.4f}"),
        (f"What was the accuracy of the {model_names[2]} model in the {experiment} benchmark?",
         f"{accuracies[2]:.4f}"),
        (f"How many models were compared in the {experiment} notebook, and which performed worst?",
         f"{len(model_names)} models, {model_names[-1]} performed worst"),
    ]

    question, answer = rng.choice(question_templates)

    # Build notebook
    config_pos = rng.randint(5, n_cells // 3)
    results_pos = rng.randint(n_cells // 3, 2 * n_cells // 3)
    winner_pos = rng.randint(2 * n_cells // 3, n_cells - 2)

    cells = []
    cells.append(f"# {experiment.replace('_', ' ').title()} — Model Comparison\n\n**Author:** {author}")

    current_len = len(cells[0])
    cell_idx = 1

    while cell_idx < n_cells and current_len < doc_length:
        if cell_idx == config_pos:
            cells.append(f"In [{cell_idx}]:\n{config_cell_code}\n\nOut [{cell_idx}]:\n{config_cell_out}")
        elif cell_idx == results_pos:
            cells.append(f"In [{cell_idx}]:\n{results_code}\n\nOut [{cell_idx}]:\n{results_out}")
        elif cell_idx == winner_pos:
            cells.append(f"In [{cell_idx}]:\n{winner_code}\n\nOut [{cell_idx}]:\n{winner_out}")
        elif rng.random() < 0.25:
            md = _generate_markdown_cell(rng, experiment.replace('_', ' '))
            cells.append(md)
        else:
            code, output = _generate_code_cell_filler(rng)
            cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")

        current_len += len(cells[-1])
        cell_idx += 1

    while current_len < doc_length:
        code, output = _generate_code_cell_filler(rng)
        cells.append(f"In [{cell_idx}]:\n{code}\n\nOut [{cell_idx}]:\n{output}")
        current_len += len(cells[-1])
        cell_idx += 1

    notebook = "\n\n---\n\n".join(cells)
    prompt = f"QUESTION: {question}\n\nNOTEBOOK:\n{notebook}"

    return NotebookQATask(
        task_id=f"notebook_cross_{task_idx}",
        prompt=prompt,
        expected_answer=answer,
        n_cells=cell_idx,
        question_type="cross_cell",
        target_cells=[config_pos, results_pos, winner_pos],
        doc_length=len(prompt),
    )


def generate_notebook_qa_suite(
    n_tasks: int = 15,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[NotebookQATask]:
    """Generate a suite of notebook QA tasks.

    Mix of task types:
    - 40% output_lookup (find specific output)
    - 30% variable_trace (trace computation across cells)
    - 30% cross_cell (cross-reference multiple cells)
    """
    if doc_lengths is None:
        doc_lengths = [15000, 25000, 40000, 60000]

    tasks = []
    rng = random.Random(42 + seed_offset)

    for i in range(n_tasks):
        doc_len = rng.choice(doc_lengths)
        n_cells = max(15, doc_len // 1000)
        seed = i + seed_offset
        r = rng.random()

        if r < 0.4:
            tasks.append(_generate_output_lookup_task(i, n_cells, doc_len, seed))
        elif r < 0.7:
            tasks.append(_generate_variable_trace_task(i, n_cells, doc_len, seed))
        else:
            tasks.append(_generate_cross_cell_task(i, n_cells, doc_len, seed))

    return tasks


def score_notebook_qa(predicted: str | None, expected: str) -> dict:
    """Score a notebook QA prediction.

    Returns dict with:
    - score: 1.0 for exact/contains match, 0.5 for partial, 0.0 otherwise
    - match_type: "exact", "contains", "partial", "none"
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    # Exact match
    if pred == exp:
        return {"score": 1.0, "match_type": "exact"}

    # Contains match (prediction contains expected)
    if exp in pred:
        return {"score": 1.0, "match_type": "contains"}

    # Reverse containment
    if pred in exp and len(pred) >= 3:
        return {"score": 0.5, "match_type": "partial"}

    # Numeric extraction — try to match numbers (with tolerance)
    import re
    pred_nums = re.findall(r'[\d.]+%?', pred)
    exp_nums = re.findall(r'[\d.]+%?', exp)
    if exp_nums and pred_nums:
        for en in exp_nums:
            if en in pred_nums:
                return {"score": 0.5, "match_type": "partial_numeric"}
            # Numeric tolerance: parse numbers and compare within 1%
            try:
                en_val = float(en.rstrip('%'))
                for pn in pred_nums:
                    pn_val = float(pn.rstrip('%'))
                    if abs(en_val - pn_val) < max(0.1, en_val * 0.01):
                        return {"score": 1.0, "match_type": "numeric_close"}
            except ValueError:
                pass

    # Key word overlap
    exp_words = set(exp.split())
    pred_words = set(pred.split())
    if len(exp_words) > 1:
        overlap = exp_words & pred_words
        if len(overlap) >= len(exp_words) * 0.7:
            return {"score": 0.5, "match_type": "partial_overlap"}

    return {"score": 0.0, "match_type": "none"}
