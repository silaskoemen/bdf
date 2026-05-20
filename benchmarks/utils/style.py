"""Shared matplotlib style and model color scheme for all benchmark plots.

JMLR-compatible: serif fonts with Computer Modern mathtext, font size 10.
Consistent color scheme across all figures by model family.

Usage:
    from benchmarks.utils.style import apply_paper_style, MODEL_COLORS, MODEL_DISPLAY_NAMES
    apply_paper_style()
"""

import matplotlib as mpl

# =============================================================================
# Matplotlib RC params (JMLR-compatible)
# =============================================================================

PAPER_RC = {
    # Font
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    # Figure
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    # Ticks
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Labels & legends
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    # Lines
    "lines.linewidth": 1.5,
    # Savefig
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def apply_paper_style() -> None:
    """Apply JMLR-compatible matplotlib style globally."""
    mpl.rcParams.update(PAPER_RC)


# =============================================================================
# Model color scheme — consistent across all figures
# =============================================================================
#
# Families:
#   BDF (proposed method)   — crimson / red
#   Tree ensembles          — green / earthy
#   Boosted models          — cool blue
#   Deep learning           — purple / pink
#   Classical / non-param.  — gray
#
# Keys cover config names, display names, and internal names so any
# plotting script can look up by whatever key it uses.

# --- Family base colors ---
_BDF = "#C03028"  # crimson (hero color)
_BDF_VARIANT = "#D95550"  # lighter crimson for BDF variants (e.g. KDE)

_RF = "#2D7D46"  # forest green
_QRF = "#5A9E6F"  # sage green
_DRF = "#3F8F58"  # mid-forest green (distinct from RF/QRF)
_BART = "#8B6914"  # dark goldenrod
_CONF_RF = "#6B8E23"  # olive drab
_CAL_RF = "#A0522D"  # sienna
_CONF_BDF = "#E08080"  # light crimson (BDF variant, conformalized)

_CONF_CATBOOST = "#2980B9"  # medium blue (conformalized CatBoost, additive)
_CQR_CATBOOST = "#7FB3D3"  # sky blue (CQR-CatBoost, adaptive)

_LGBM = "#2E86AB"  # steel blue
_NGBOOST = "#4169E1"  # royal blue
_XGBLSS = "#1F77B4"  # xgboost blue
_CATBOOST = "#1B4F72"  # dark navy
_CONF_LGBM = "#5DADE2"  # light blue
_LGBM_QREG = "#6BAED6"  # sky blue

_GAUSSIAN_DE = "#7B2D8E"  # deep purple
_TREEFFUSER = "#C06DB0"  # orchid

_GP = "#555555"  # dark gray
_BAYES_RIDGE = "#888888"  # medium gray
_KNN = "#333333"  # charcoal

# --- Unified lookup (all possible keys → color) ---
MODEL_COLORS: dict[str, str] = {
    # BDF family
    "BDF": _BDF,
    "bdf": _BDF,
    "BDFNormal": _BDF,
    "BDF (Normal)": _BDF,
    "bdf_normalmunormal": _BDF,
    "BDFKDE": _BDF_VARIANT,
    "BDF (KDE)": _BDF_VARIANT,
    "bdf_kde": _BDF_VARIANT,
    "ConfBDFNormal": _CONF_BDF,
    "ConfBDFKDE": "#E8A0A0",
    "bdf_betamvbernoulli": _BDF,
    # Tree ensembles
    "RandomForest": _RF,
    "Random Forest": _RF,
    "RF": _RF,
    "rf_reg": _RF,
    "rf_clas": _RF,
    "QRF": _QRF,
    "qrf": _QRF,
    "DRF": _DRF,
    "drf": _DRF,
    "BART": _BART,
    "bartpy": _BART,
    "pymc_bart": _BART,
    "ConfRF": _CONF_RF,
    "ConformalRF": _CONF_RF,
    "Conformal RF": _CONF_RF,
    "confrf": _CONF_RF,
    "CalRF": _CAL_RF,
    "calrf_clas": _CAL_RF,
    # Boosted models
    "LightGBM": _LGBM,
    "lgbm_reg": _LGBM,
    "lgbm_clas": _LGBM,
    "lgbm_qreg": _LGBM_QREG,
    "NGBoost": _NGBOOST,
    "ngboost_reg": _NGBOOST,
    "ngboost_clas": _NGBOOST,
    "XGBoostLSS": _XGBLSS,
    "xgboostlss_gaussian": _XGBLSS,
    "xgboostlss_studentt": _XGBLSS,
    "xgboostlss_laplace": _XGBLSS,
    "xgboostlss_gaussian_mixture": _XGBLSS,
    "CatBoostUnc": _CATBOOST,
    "CatBoostUncertainty": _CATBOOST,
    "catbunc_reg": _CATBOOST,
    "ConfCatBoost": _CONF_CATBOOST,
    "CQRCatBoost": _CQR_CATBOOST,
    "ConfLGBM": _CONF_LGBM,
    "conflgbm": _CONF_LGBM,
    # Deep learning / generative
    "GaussianDE": _GAUSSIAN_DE,
    "gaussian_de": _GAUSSIAN_DE,
    "Treeffuser": _TREEFFUSER,
    "treeffuser": _TREEFFUSER,
    # Classical / non-parametric
    "GP": _GP,
    "gp_reg": _GP,
    "gp_clas": _GP,
    "GaussianProcess": _GP,
    "BayesRidge": _BAYES_RIDGE,
    "bayesridge_reg": _BAYES_RIDGE,
    "KNNKDE": _KNN,
    "knnkde": _KNN,
    "KNN": _KNN,
    "knn_clas": _KNN,
}

# --- Display names (config key → paper-friendly label) ---
MODEL_DISPLAY_NAMES: dict[str, str] = {
    # BDF
    "BDF": "BDF",
    "bdf_normalmunormal": "BDF",
    "bdf_betamvbernoulli": "BDF",
    "bdf_kde": "BDF (KDE)",
    "BDFNormal": "BDF (Normal)",
    "BDFKDE": "BDF (KDE)",
    "ConfBDFNormal": "Conf-BDF (Normal)",
    "ConfBDFKDE": "Conf-BDF (KDE)",
    # Tree ensembles
    "RandomForest": "Random Forest",
    "rf_reg": "RF",
    "rf_clas": "RF",
    "QRF": "QRF",
    "qrf": "QRF",
    "DRF": "DRF",
    "drf": "DRF",
    "bartpy": "BART",
    "pymc_bart": "BART",
    "confrf": "ConfRF",
    "ConformalRF": "Conformal RF",
    "calrf_clas": "CalRF",
    # Boosted
    "lgbm_reg": "LightGBM",
    "lgbm_clas": "LightGBM",
    "lgbm_qreg": "LGBM-QR",
    "ngboost_reg": "NGBoost",
    "ngboost_clas": "NGBoost",
    "XGBoostLSS": "XGBoostLSS",
    "xgboostlss_gaussian": "XGBLSS-Gauss",
    "xgboostlss_studentt": "XGBLSS-t",
    "xgboostlss_laplace": "XGBLSS-Laplace",
    "xgboostlss_gaussian_mixture": "XGBLSS-GMix",
    "catbunc_reg": "CatBoostUnc",
    "CatBoostUncertainty": "CatBoost (Unc)",
    "ConfCatBoost": "Conf-CatBoost",
    "CQRCatBoost": "CQR-CatBoost",
    "conflgbm": "ConfLGBM",
    # Deep learning
    "gaussian_de": "GaussianDE",
    "treeffuser": "Treeffuser",
    # Classical
    "gp_reg": "GP",
    "gp_clas": "GP",
    "bayesridge_reg": "BayesRidge",
    "knnkde": "KNNKDE",
    "knn_clas": "KNN",
}

# --- Markers (for line plots needing shape differentiation) ---
MODEL_MARKERS: dict[str, str] = {
    "BDF": "o",
    "BDFNormal": "o",
    "BDFKDE": "D",
    "RandomForest": "s",
    "RF": "s",
    "QRF": "^",
    "DRF": "v",
    "BART": "v",
    "ConfRF": "<",
    "ConformalRF": "<",
    "CalRF": ">",
    "LightGBM": "P",
    "NGBoost": "X",
    "CatBoostUnc": "h",
    "CatBoostUncertainty": "h",
    "ConfCatBoost": "H",
    "CQRCatBoost": "8",
    "ConfBDFNormal": "d",
    "ConfBDFKDE": "D",
    "ConfLGBM": "d",
    "GaussianDE": "*",
    "Treeffuser": "p",
    "GP": "+",
    "BayesRidge": "x",
    "KNNKDE": "1",
    "KNN": "2",
}

# --- General-purpose semantic colors (for non-model elements) ---
SEMANTIC_COLORS = {
    "ground_truth": "#2E86AB",  # steel blue
    "bdf": _BDF,  # crimson (matches model color)
    "baseline": "#F18F01",  # orange
    "data": "#4A4A4A",  # dark gray
    "confidence": "#C73E1D",  # burnt sienna
    "reference": "#999999",  # medium gray
}


def get_color(key: str) -> str:
    """Look up model color, falling back to gray for unknown models."""
    return MODEL_COLORS.get(key, "#AAAAAA")


def get_display_name(key: str) -> str:
    """Look up display name, falling back to key itself."""
    return MODEL_DISPLAY_NAMES.get(key, key)
