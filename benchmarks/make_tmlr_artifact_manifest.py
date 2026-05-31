"""Write a compact manifest for the TMLR submission artifacts.

The manifest is intentionally generated from local files only. It records the
paper build inputs from ``paper/main.fls`` plus the result YAMLs consumed by the
locked real-data benchmark aggregators and revision tables.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "benchmarks/results/TMLR_ARTIFACT_MANIFEST.md"
PAPER_FLS = ROOT / "paper/main.fls"

REGRESSION_DATASETS = [
    "abalone_age",
    "bike_sharing",
    "boston_housing",
    "california_housing",
    "combined_cycle_power_plant",
    "concrete_strength",
    "energy_efficiency",
    "forest_fires",
    "kin8nm",
    "naval_propulsion",
    "parkinsons_updrs",
    "protein_tertiary_structure",
    "superconductor",
    "wine_quality",
    "yacht_hydrodynamics",
]

CLASSIFICATION_DATASETS = [
    "ionosphere",
    "breast_cancer_wisconsin",
    "credit_approval",
    "heart_disease",
    "pima_diabetes",
    "titanic",
    "german_credit",
    "aids_ctg_175",
    "spambase",
    "default_credit_card",
    "bank_marketing",
    "adult_income",
]

REGRESSION_RESULT_MODELS = [
    "bayesridge_reg",
    "bdf_freqstudentt",
    "bdf_freqstudentt_nll",
    "bdf_freqstudentt_nll_bic",
    "bdf_gammamvlambdapoisson",
    "bdf_kde",
    "bdf_normalmunormal",
    "bdf_normalmunormal_nle",
    "bdf_normalmunormal_nll",
    "bdf_normalmunormal_nll_bic",
    "catbunc_reg",
    "climatological",
    "conflgbm",
    "confrf",
    "drf",
    "gaussian_de",
    "knnkde",
    "lgbm_reg",
    "ngboost_reg",
    "qrf",
    "xgboostlss_gaussian",
    "xgboostlss_gaussian_mixture",
    "xgboostlss_laplace",
    "xgboostlss_studentt",
]

CLASSIFICATION_RESULT_MODELS = [
    "bdf_betamvbernoulli",
    "bdf_betamvbernoulli_nle",
    "calrf_clas",
    "callgbm_clas",
    "knn_clas",
    "lgbm_clas",
    "ngboost_clas",
    "rf_clas",
]

REGENERATION_COMMANDS = [
    "pixi run fetch-data",
    "pixi run process-data",
    "pixi run reg-suite",
    "pixi run -e benchmark reg-suite-models",
    "pixi run clas-suite",
    "pixi run -e benchmark clas-suite-models",
    "pixi run python -m benchmarks.make_revision_bdf_tables",
    "pixi run reg",
    "pixi run clas",
    "pixi run synth-study",
    "pixi run -e benchmark synth-study-models --models QRF --result-suffix forests",
    "pixi run -e benchmark synth-study-models --models DRF --result-suffix drf",
    "pixi run synth-output",
    "pixi run paper",
    "pixi run tmlr-manifest",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"
    return result.stdout.strip() or "clean"


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _paper_inputs() -> list[Path]:
    if not PAPER_FLS.exists():
        return []

    inputs: set[Path] = set()
    for line in PAPER_FLS.read_text(errors="ignore").splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = line.removeprefix("INPUT ").strip()
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / "paper" / path).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.exists() and path.is_file():
            inputs.add(path)
    return sorted(inputs, key=_rel)


def _result_paths() -> tuple[list[Path], list[Path]]:
    regression = [ROOT / f"benchmarks/results/regression/res_{model}.yaml" for model in REGRESSION_RESULT_MODELS]
    classification = [
        ROOT / f"benchmarks/results/classification/res_{model}.yaml" for model in CLASSIFICATION_RESULT_MODELS
    ]
    return sorted((path for path in regression if path.exists()), key=_rel), sorted(
        (path for path in classification if path.exists()), key=_rel
    )


def _paper_generated_tables(paper_inputs: list[Path]) -> list[Path]:
    return sorted(
        (
            path
            for path in paper_inputs
            if path.suffix == ".tex"
            and "benchmarks" in path.parts
            and ("tables" in path.parts or "results" in path.parts)
        ),
        key=_rel,
    )


def _paper_generated_plots(paper_inputs: list[Path]) -> list[Path]:
    return sorted((path for path in paper_inputs if path.suffix == ".pdf" and "plots" in path.parts), key=_rel)


def _markdown_file_table(paths: list[Path]) -> list[str]:
    if not paths:
        return ["_None found._", ""]
    lines = ["| Path | SHA-256 |", "|---|---|"]
    lines.extend(f"| `{_rel(path)}` | `{_sha256(path)}` |" for path in paths)
    lines.append("")
    return lines


def main() -> None:
    paper_pdf = ROOT / "paper/main.pdf"
    regression_results, classification_results = _result_paths()
    paper_inputs = _paper_inputs()
    table_paths = _paper_generated_tables(paper_inputs)
    plot_paths = _paper_generated_plots(paper_inputs)

    lines = [
        "# TMLR Artifact Manifest",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Git commit: `{_git(['rev-parse', 'HEAD'])}`",
        f"Git status: `{_git(['status', '--short'])}`",
        "",
        "## Protocol Lock",
        "",
        "- Real-data seed: `1234`.",
        "- Real-data splitters: regression uses `KFold`, classification uses `StratifiedKFold`.",
        "- Fold convention: 10 shuffled folds; fold 0 is tuning only, folds 1--9 are evaluation folds.",
        "- Tuning metrics: CRPS for regression, log-loss for classification.",
        "- Main synthetic DGP seed: `42`, with the same fold convention.",
        "- Result YAMLs record the run command, git commit, dirty flag, seed, fold count, tuning metric, "
        "resolved model config, best parameters, per-fold metrics, and dataset metadata.",
        "",
        "Regression datasets: " + ", ".join(f"`{name}`" for name in REGRESSION_DATASETS) + ".",
        "",
        "Classification datasets: " + ", ".join(f"`{name}`" for name in CLASSIFICATION_DATASETS) + ".",
        "",
        "## Regeneration Commands",
        "",
        "Run these from the repository root, after installing dependencies with `pixi install`.",
        "",
        "```bash",
        *REGENERATION_COMMANDS,
        "```",
        "",
        "## Paper PDF",
        "",
    ]
    lines.extend(_markdown_file_table([paper_pdf] if paper_pdf.exists() else []))

    lines.extend(["## Locked Real-Data Result YAMLs", "", "### Regression", ""])
    lines.extend(_markdown_file_table(regression_results))
    lines.extend(["### Classification", ""])
    lines.extend(_markdown_file_table(classification_results))

    lines.extend(["## Generated Tables Consumed By The Paper", ""])
    lines.extend(_markdown_file_table(table_paths))

    lines.extend(["## Generated Plot PDFs Consumed By The Paper", ""])
    lines.extend(_markdown_file_table(plot_paths))

    lines.extend(
        [
            "## Paper Build Inputs",
            "",
            "These are repository-local files listed as `INPUT` entries in `paper/main.fls`. "
            "They include manuscript sources, generated table snippets, and plot PDFs consumed by LaTeX.",
            "",
        ]
    )
    lines.extend(_markdown_file_table(paper_inputs))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {_rel(OUT_PATH)}")


if __name__ == "__main__":
    main()
