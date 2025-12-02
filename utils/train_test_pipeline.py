import optuna
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from typing import Tuple, Dict, Any, Optional


# Configuration
@dataclass
class ExperimentConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 100
    cv_folds: int = 3
    tfidf_max_features: int = 100000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    solver: str = "liblinear"
    penalty: str = "l2"


# Objective Function
class OptunaObjective:
    """
    Callable class for Optuna to optimize hyperparameters.
    Keeps the optimization logic separate from the training pipeline.
    """

    def __init__(self, X, y, config: ExperimentConfig):
        self.X = X
        self.y = y
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        # 1. Suggest Hyperparameters
        tfidf_min_df = trial.suggest_int("tfidf__min_df", 2, 10)
        tfidf_max_df = trial.suggest_float("tfidf__max_df", 0.6, 0.9)
        clf_c = trial.suggest_float("clf__C", 8, 15, log=True)
        clf_tol = trial.suggest_float("clf__tol", 1e-5, 1e-3, log=True)

        # 2. Build Pipeline
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.config.tfidf_ngram_range,
                        max_features=self.config.tfidf_max_features,
                        min_df=tfidf_min_df,
                        max_df=tfidf_max_df,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        solver=self.config.solver,
                        penalty=self.config.penalty,
                        C=clf_c,
                        tol=clf_tol,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        # 3. Evaluate
        scores = cross_val_score(
            pipeline,
            self.X,
            self.y,
            n_jobs=-1,
            cv=self.config.cv_folds,
            scoring="accuracy",
        )
        return scores.mean()


# Helper Functions
def plot_results(y_true, y_pred, classes, title_suffix=""):
    """Generates and plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.grid(False)
    plt.show()


def build_final_pipeline(params: Dict[str, Any], config: ExperimentConfig) -> Pipeline:
    """Reconstructs the pipeline with winning parameters."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.tfidf_ngram_range,
                    max_features=config.tfidf_max_features,
                    min_df=params["tfidf__min_df"],
                    max_df=params["tfidf__max_df"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver=config.solver,
                    penalty=config.penalty,
                    C=params["clf__C"],
                    tol=params["clf__tol"],
                    random_state=config.random_state,
                ),
            ),
        ]
    )


# Main Workflow
def train_and_evaluate(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    config: Optional[ExperimentConfig] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:

    if config is None:
        config = ExperimentConfig()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting Experiment with {config.n_trials} trials...")

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[target_col],
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[target_col],
    )

    # 2. Run Optimization
    objective = OptunaObjective(X_train, y_train, config)
    study = optuna.create_study(direction="maximize")

    # Using tqdm context manager for cleaner output
    with tqdm(total=config.n_trials, desc="Optimizing Hyperparameters") as pbar:
        # Callback to update pbar
        def progress_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"best_score": f"{study.best_value:.4f}"})

        study.optimize(
            objective, n_trials=config.n_trials, callbacks=[progress_callback]
        )

    print(f"\nBest CV Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")

    # 3. Retrain Final Model
    print("Retraining final model on full training set...")
    final_pipeline = build_final_pipeline(study.best_params, config)
    final_pipeline.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = final_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 30)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 5. Visualize
    plot_results(
        y_test,
        y_pred,
        final_pipeline.classes_,
        title_suffix=f"(Optuna C={study.best_params['clf__C']:.2f})",
    )

    try:
        # Optuna's matplotlib backend returns an Axes object
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.show()
    except ImportError:
        pass
    except Exception as e:
        print(f"Could not plot importance: {e}")

    return final_pipeline, {"accuracy": accuracy, "y_test": y_test, "y_pred": y_pred}
