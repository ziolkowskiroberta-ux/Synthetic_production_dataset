import os
# Suppress ConvergenceWarnings from joblib parallel workers (DMLOrthoForest /
# uplift-forest uses n_jobs=-1; child processes are spawned on Windows and do
# NOT inherit the parent's warnings.filterwarnings() state, but they DO inherit
# environment variables set before the first worker is spawned).
# The warnings are benign: Duality gap = 0.0 means the optimum was reached,
# but sklearn's convergence check fails on exact-zero comparisons.
os.environ["PYTHONWARNINGS"] = "ignore"

import random
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseTRegressor, BaseXRegressor, BaseRRegressor, BaseDRRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

try:
    from econml.dml import CausalForestDML
    from econml.orf import DMLOrthoForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

LINE_COLORS = {
    1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a', 4: '#984ea3', 5: '#ff7f00',
    6: '#ffff33', 7: '#a65628', 8: '#f781bf', 9: '#999999', 10: '#66c2a5'
}

CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
SEED = 42
N_BOOTSTRAP = 999
N_PLACEBO   = 10

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data(filepath='synthetic_production_dataset.csv'):
    df = pd.read_csv(filepath)
    print("=" * 80)
    print("DATASET LOADED")
    print("=" * 80)
    print(f"Total observations: {len(df)}")
    print(f"Treatment: {df['treatment'].sum()} treated, {len(df) - df['treatment'].sum()} control")
    return df

def prepare_features(df):
    features = ['order_size', 'num_operations', 'operators', 'shifts',
                'setup_time', 'defect_rate', 'throughput']
    X = df[features].copy()
    dummies = pd.get_dummies(df['product_type'], prefix='prod')
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    X = pd.concat([X, dummies], axis=1)
    return X

# ============================================================================
# CAUSAL INFERENCE - META-LEARNERS
# ============================================================================

def get_base_learner(model_type):
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=100, random_state=SEED, max_depth=10),
        'gb': GradientBoostingRegressor(n_estimators=100, random_state=SEED, max_depth=5)
    }
    if model_type not in models:
        raise ValueError(f"Model '{model_type}' not recognized.")
    return models[model_type]

def get_meta_learner(learner_type, base_learner):
    learners = {
        't-learner': BaseTRegressor(learner=base_learner),
        'x-learner': BaseXRegressor(learner=base_learner),
        'r-learner': BaseRRegressor(learner=base_learner),
        'dr-learner': BaseDRRegressor(learner=base_learner)
    }
    if learner_type not in learners:
        raise ValueError(f"Learner '{learner_type}' not recognized.")
    return learners[learner_type]

def estimate_ite_metalearner(df, outcome, learner_type='t-learner', base_model='linear'):
    print(f"\nFitting {learner_type.upper()} with {base_model.upper()} for outcome: {outcome}...")
    np.random.seed(SEED)
    X = prepare_features(df)
    T = df['treatment']
    y = df[outcome]
    base_learner = get_base_learner(base_model)
    learner = get_meta_learner(learner_type, base_learner)
    learner.fit(X=X, treatment=T, y=y)
    if learner_type == 'r-learner':
        ite = learner.predict(X=X)
    else:
        ite = learner.predict(X=X, treatment=None)
    df_result = df.copy()
    df_result['ite'] = ite.flatten() if hasattr(ite, 'flatten') else ite
    return df_result, learner

# ============================================================================
# CAUSAL INFERENCE - TREE-BASED METHODS
# ============================================================================

def estimate_ite_causal_forest(df, outcome, n_estimators=100, max_depth=10, min_samples_leaf=50):
    if not ECONML_AVAILABLE:
        raise ImportError("econml required")
    print(f"\nFitting CAUSAL FOREST for {outcome}...")
    X = prepare_features(df)
    T = df['treatment'].values.reshape(-1, 1)
    y = df[outcome].values
    cf_model = CausalForestDML(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, random_state=SEED,
                               verbose=0, n_jobs=-1)
    cf_model.fit(Y=y, T=T, X=X)
    ite = cf_model.effect(X=X)
    df_result = df.copy()
    df_result['ite'] = ite.flatten() if hasattr(ite, 'flatten') else ite
    return df_result, cf_model

def estimate_ite_uplift_forest(df, outcome, n_estimators=100, max_depth=10, min_samples_leaf=50):
    if not ECONML_AVAILABLE:
        raise ImportError("econml required")
    print(f"\nFitting ORTHOGONAL RF for {outcome}...")
    X = prepare_features(df)
    T = df['treatment'].values.reshape(-1, 1)
    y = df[outcome].values
    # Replace default LassoCV (5-fold internal CV per tree) with a fixed-alpha
    # Lasso to eliminate the ~5x CV overhead.  alpha=0.1 is a conservative default
    _nuisance = Lasso(alpha=0.1, max_iter=10000, warm_start=True)
    orf_model = DMLOrthoForest(n_trees=n_estimators, max_depth=max_depth,
                               min_leaf_size=min_samples_leaf,
                               model_Y=_nuisance, model_T=_nuisance,
                               random_state=SEED, n_jobs=-1, verbose=0)
    orf_model.fit(Y=y, T=T, X=X)
    ite = orf_model.effect(X=X)
    df_result = df.copy()
    df_result['ite'] = ite.flatten() if hasattr(ite, 'flatten') else ite
    return df_result, orf_model

def estimate_ite_interaction_tree(df, outcome, max_depth=5, min_samples_leaf=100):
    print(f"\nFitting INTERACTION TREE for {outcome}...")
    X = prepare_features(df)
    T = df['treatment']
    y = df[outcome]
    X_augmented = X.copy()
    X_augmented['treatment'] = T.values
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED)
    tree.fit(X_augmented, y)
    X_treated = X.copy()
    X_treated['treatment'] = 1
    y1_pred = tree.predict(X_treated)
    X_control = X.copy()
    X_control['treatment'] = 0
    y0_pred = tree.predict(X_control)
    ite = y1_pred - y0_pred
    df_result = df.copy()
    df_result['ite'] = ite
    return df_result, tree

def estimate_ite_interaction_forest(df, outcome, n_estimators=100, max_depth=5, min_samples_leaf=100):
    print(f"\nFitting INTERACTION FOREST for {outcome}...")
    X = prepare_features(df)
    T = df['treatment']
    y = df[outcome]
    X_augmented = X.copy()
    X_augmented['treatment'] = T.values
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, max_features='sqrt',
                               random_state=SEED, n_jobs=-1)
    rf.fit(X_augmented, y)
    X_treated = X.copy()
    X_treated['treatment'] = 1
    y1_pred = rf.predict(X_treated)
    X_control = X.copy()
    X_control['treatment'] = 0
    y0_pred = rf.predict(X_control)
    ite = y1_pred - y0_pred
    df_result = df.copy()
    df_result['ite'] = ite
    return df_result, rf

# ============================================================================
# UNIFIED ESTIMATION FUNCTION
# ============================================================================

def estimate_ite(df, outcome, method='dr-learner', base_model='rf', **kwargs):
    if method in ['t-learner', 'x-learner', 'r-learner', 'dr-learner']:
        return estimate_ite_metalearner(df, outcome, method, base_model)
    elif method == 'causal-forest':
        return estimate_ite_causal_forest(df, outcome, **kwargs)
    elif method == 'uplift-forest':
        return estimate_ite_uplift_forest(df, outcome, **kwargs)
    elif method == 'interaction-tree':
        return estimate_ite_interaction_tree(df, outcome, **kwargs)
    elif method == 'interaction-forest':
        return estimate_ite_interaction_forest(df, outcome, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not recognized.")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE_LEVEL, seed=SEED):
    """
    Percentile bootstrap CI for the mean of values (clustered resampling).

    Resamples observations with replacement at the level of the provided array.
    For clustered bootstrap at a higher level (e.g., products within a line),
    pass the cluster-level means as input.

    Returns
    -------
    (mean, ci_low, ci_high) : tuple of floats
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    half_alpha = (1.0 - confidence) / 2.0
    ci_low  = float(np.percentile(boot_means, half_alpha * 100.0))
    ci_high = float(np.percentile(boot_means, (1.0 - half_alpha) * 100.0))
    return float(values.mean()), ci_low, ci_high

# ============================================================================
# AGGREGATION AND META-ANALYSIS
# ============================================================================

def aggregate_by_line_product(df, outcome):
    """
    Aggregate ITE estimates by (line, product) group using bootstrap CI (95%).

    Heterogeneity is measured by tau (SD of individual ITEs within the group),
    which quantifies the spread of individual treatment effects.

    Returns
    -------
    DataFrame with columns:
        line_id, product_type, line_product, n,
        mean, se, ci_low, ci_high, tau, tau2
    """
    df = df.copy()
    df['line_product'] = df['line_id'].astype(str) + '_' + df['product_type']
    results = []
    z_crit = stats.norm.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0)

    for name, group in df.groupby('line_product'):
        ite_values = group['ite'].values
        n = len(ite_values)
        if n < 2:
            continue

        # Bootstrap CI for the group mean (clustered within line-product)
        mean, ci_low, ci_high = bootstrap_ci(ite_values)

        # SE derived from bootstrap CI width (used downstream for inverse-variance)
        ci_width = ci_high - ci_low
        se = ci_width / (2.0 * z_crit) if ci_width > 0 else 1e-8

        # Heterogeneity of individual ITEs within the group
        tau2 = float(np.var(ite_values, ddof=1))
        tau  = float(np.sqrt(tau2))

        line_id = int(name.split('_')[0])
        product = name.split('_')[1]
        results.append({
            'line_id':      line_id,
            'product_type': product,
            'line_product': name,
            'n':            n,
            'mean':         mean,
            'se':           se,
            'ci_low':       ci_low,
            'ci_high':      ci_high,
            'tau':          tau,
            'tau2':         tau2,
        })
    return pd.DataFrame(results)


def meta_analysis_by_line(df_agg):
    """
    Pool line-product ITE estimates at line level using inverse-variance weighting.

    CI is computed via bootstrap (resampling line-product groups within each line).
    Heterogeneity (I²) is computed from Cochran's Q across products.

    Parameters
    ----------
    df_agg : DataFrame output of aggregate_by_line_product()

    Returns
    -------
    DataFrame with columns:
        line_id, n_products, pooled_mean, ci_low, ci_high, I2, Q
    """
    results = []
    for line_id in sorted(df_agg['line_id'].unique()):
        line_data = df_agg[df_agg['line_id'] == line_id].reset_index(drop=True)
        k = len(line_data)
        if k < 2:
            continue

        means   = line_data['mean'].values
        ses     = line_data['se'].values
        weights = 1.0 / (ses ** 2)

        # Inverse-variance pooled mean
        pooled_mean = float(np.sum(weights * means) / np.sum(weights))

        # Bootstrap CI: resample line-product groups (clusters) within the line
        rng = np.random.default_rng(SEED)
        indices = np.arange(k)
        boot_pool_means = []
        for _ in range(N_BOOTSTRAP):
            idx      = rng.choice(indices, size=k, replace=True)
            b_means  = means[idx]
            b_ses    = ses[idx]
            b_w      = 1.0 / (b_ses ** 2)
            boot_pool_means.append(float(np.sum(b_w * b_means) / np.sum(b_w)))

        boot_pool_means = np.array(boot_pool_means)
        half_alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
        ci_low  = float(np.percentile(boot_pool_means, half_alpha * 100.0))
        ci_high = float(np.percentile(boot_pool_means, (1.0 - half_alpha) * 100.0))

        # Heterogeneity: Cochran's Q and I²
        Q   = float(np.sum(weights * (means - pooled_mean) ** 2))
        df_q = k - 1
        I2  = max(0.0, (Q - df_q) / Q * 100.0) if Q > 0 else 0.0

        results.append({
            'line_id':     line_id,
            'n_products':  k,
            'pooled_mean': pooled_mean,
            'ci_low':      ci_low,
            'ci_high':     ci_high,
            'I2':          I2,
            'Q':           Q,
        })
    return pd.DataFrame(results)

# ============================================================================
# VALIDATION (PLACEBO TEST)
# ============================================================================

def validate_model(df, outcome, method, base_model='rf',
                   confidence=CONFIDENCE_LEVEL, seed=SEED,
                   n_placebo=N_PLACEBO, **kwargs):
    """
    Permutation-based placebo test.

    Runs n_placebo independent treatment randomizations. For each run the model
    is fitted on the permuted data and the overall mean ITE across all
    line-product groups is recorded. The CI is the percentile interval of this
    permutation distribution.

    This correctly accounts for between-randomization variability and avoids
    the over-precision artefact of a single-randomization bootstrap.

    Pass criterion : 95% permutation CI contains zero.
    D statistic    : |median of permutation means| / CI_width
                     (smaller D → permutation distribution more centred at zero)

    Returns
    -------
    dict with keys:
        placebo_mean, ci_low, ci_high, is_valid, d_stat,
        permutation_means, n_placebo
    """
    print(f"  Placebo ({outcome}): {n_placebo} permutations ...", end=" ", flush=True)
    rng = np.random.default_rng(seed)
    treatment_prop = df['treatment'].mean()

    perm_means = []
    for _ in range(n_placebo):
        df_perm = df.copy()
        df_perm['treatment'] = rng.binomial(1, treatment_prop, size=len(df))
        df_perm_ite, _ = estimate_ite(df_perm, outcome, method, base_model, **kwargs)
        agg_perm = aggregate_by_line_product(df_perm_ite, outcome)
        perm_means.append(float(agg_perm['mean'].mean()))

    perm_means = np.array(perm_means)
    print("done")

    half_alpha  = (1.0 - confidence) / 2.0
    ci_low  = float(np.percentile(perm_means, half_alpha * 100.0))
    ci_high = float(np.percentile(perm_means, (1.0 - half_alpha) * 100.0))
    placebo_mean = float(np.median(perm_means))

    is_valid = bool(ci_low <= 0.0 <= ci_high)

    ci_width = ci_high - ci_low
    d_stat   = float(abs(placebo_mean) / ci_width) if ci_width > 0 else np.inf

    return {
        'placebo_mean':     placebo_mean,
        'ci_low':           ci_low,
        'ci_high':          ci_high,
        'is_valid':         is_valid,
        'd_stat':           d_stat,
        'permutation_means': perm_means,
        'n_placebo':        n_placebo,
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

OUTCOME_LABELS = {'efficiency': 'Efficiency', 'wip': 'WIP', 'leadtime': 'Lead Time'}

def format_outcome(outcome):
    return OUTCOME_LABELS.get(outcome.lower(), outcome.capitalize())

def apply_plot_style(fig, ax):
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#EBEBEB')


# ----------------------------------------------------------------------------
# Shared drawing primitive — keeps all plots visually consistent
# Per-plot whisker sizing passed as keyword arguments.
# ----------------------------------------------------------------------------
_DOT_SIZE   = 100   # circle area (scatter s=) — placebo plot
_BAR_LW     = 2.5   # horizontal CI bar linewidth
_ZERO_LW    = 2.2   # zero-reference vertical line linewidth

# Combined-plot specific sizing (narrower panels need proportionally smaller elements)
_DOT_SIZE_COMB     = 55    # smaller circle for narrower combined panels
_WHISKER_H_COMB_LP = 0.20  # whisker half-height for line-product combined panel
_WHISKER_LW_COMB_LP = 1.9  # whisker linewidth for line-product combined panel
_WHISKER_H_COMB_L  = 0.12  # whisker half-height for line combined panel
_WHISKER_LW_COMB_L = 1.6   # whisker linewidth for line combined panel
_BAR_LW_COMB       = 2.0   # horizontal bar linewidth for combined panels


def _draw_ci_row(ax, y, mean, ci_low, ci_high, color,
                 whisker_h=0.22, whisker_lw=2.2,
                 dot_size=None, bar_lw=None):
    """
    Standardized CI row (same symbology across all plots):
      ├── horizontal line spanning [ci_low, ci_high]
      ├── vertical tip whiskers at ci_low and ci_high
      └── filled circle (○) at mean
    whisker_h  : half-height of the vertical tip in y-axis units
    whisker_lw : linewidth of the vertical tip
    dot_size   : scatter marker area (defaults to _DOT_SIZE)
    bar_lw     : horizontal bar linewidth (defaults to _BAR_LW)
    """
    _s   = dot_size if dot_size is not None else _DOT_SIZE
    _blw = bar_lw  if bar_lw  is not None else _BAR_LW

    # Horizontal bar
    ax.plot([ci_low, ci_high], [y, y],
            color=color, linewidth=_blw, alpha=0.85,
            solid_capstyle='butt', zorder=3)
    # Vertical tip whiskers
    for xv in (ci_low, ci_high):
        ax.plot([xv, xv], [y - whisker_h, y + whisker_h],
                color=color, linewidth=whisker_lw, zorder=3)
    # Circle for point estimate
    ax.scatter(mean, y, color=color, s=_s, zorder=5,
               edgecolors='black', linewidth=0.8, marker='o')


def _xlim_with_margin(ax, all_values, margin_frac=0.12):
    """Set x-axis limits with symmetric margin, always including zero."""
    vals      = np.concatenate([np.asarray(all_values).ravel(), [0.0]])
    data_min  = vals.min()
    data_max  = vals.max()
    margin    = (data_max - data_min) * margin_frac
    ax.set_xlim([data_min - margin, data_max + margin])


# ----------------------------------------------------------------------------
# Plot 1 — Placebo test (all outcomes combined)
# ----------------------------------------------------------------------------

def plot_validation(validation_results_dict, method_name=""):
    """
    Placebo test: point estimate + Bootstrap 95% CI for all outcomes in one figure.
    Each outcome is a row. Legend encodes mean, CI bounds, D statistic and PASS/FAIL.
    """
    outcomes  = list(validation_results_dict.keys())
    n         = len(outcomes)
    colors_oc = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, ax = plt.subplots(figsize=(11, max(4, n * 1.7 + 1.2)))
    apply_plot_style(fig, ax)

    for i, outcome in enumerate(outcomes):
        res       = validation_results_dict[outcome]
        y         = n - i - 1
        color     = colors_oc[i % len(colors_oc)]
        valid_str = 'PASS' if res['is_valid'] else 'FAIL'
        label = (f"{format_outcome(outcome)}: "
                 f"\u03bc={res['placebo_mean']:.4f}  "
                 f"CI=[{res['ci_low']:.4f}, {res['ci_high']:.4f}]  "
                 f"D={res['d_stat']:.4f}  [{valid_str}]")

        _draw_ci_row(ax, y, res['placebo_mean'], res['ci_low'], res['ci_high'], color)
        ax.scatter([], [], color=color, s=_DOT_SIZE, marker='o',
                   edgecolors='black', linewidth=0.9, label=label)

    ax.axvline(0, color='red', linestyle='--', linewidth=_ZERO_LW,
               alpha=0.9, zorder=4, label='H\u2080: zero effect (placebo criterion)')

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([format_outcome(o) for o in outcomes][::-1],
                       fontsize=12, fontweight='bold')
    ax.set_xlabel('Permutation Mean ITE  (95% Percentile CI)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        f"{method_name} | Placebo Test\n"
        f"Permutation CI  \u00b7  D = |median| / CI width  \u00b7  n={list(validation_results_dict.values())[0]['n_placebo']} permutations",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9)
    ax.grid(axis='x', alpha=0.35, linewidth=0.8)
    all_ci = [[r['ci_low'], r['ci_high']] for r in validation_results_dict.values()]
    _xlim_with_margin(ax, all_ci)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Plot 2 — ITE by Line-Product (all outcomes side by side)
# ----------------------------------------------------------------------------

def plot_ite_ci_combined(results_by_outcome, method_name=""):
    """
    Combined 3-panel figure: ITE by Line-Product for all outcomes side by side.

    Y-axis labels (line-product names) appear on the leftmost panel only.
    τ values differ per outcome so are omitted from shared y-labels; the
    suptitle notes that τ = SD of individual ITEs.
    """
    outcomes = list(results_by_outcome.keys())
    n_out    = len(outcomes)

    # Determine row count from first outcome (same groups across all outcomes)
    first_df = results_by_outcome[outcomes[0]].sort_values(
        ['line_id', 'product_type']).reset_index(drop=True)
    n = len(first_df)

    # Simplified y-labels: line-product only (no τ — it differs per outcome)
    y_labels = [f"{r['line_product']}" for _, r in first_df.iterrows()][::-1]

    fig_w = 7.5 * n_out
    fig_h = max(8.0, n * 0.54 + 2.5)
    # sharey=False: avoid FixedFormatter-sharing bug where set_yticklabels([])
    # on an inner axis clears the shared formatter and erases all y-labels.
    # Y-limits are synchronised manually below.
    fig, axes = plt.subplots(1, n_out, figsize=(fig_w, fig_h))
    if n_out == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    for col_idx, (outcome, ax) in enumerate(zip(outcomes, axes)):
        ax.set_facecolor('#EBEBEB')
        df_s = results_by_outcome[outcome].sort_values(
            ['line_id', 'product_type']).reset_index(drop=True)

        for i, row in df_s.iterrows():
            y     = n - i - 1
            color = LINE_COLORS.get(int(row['line_id']), 'gray')
            _draw_ci_row(ax, y, row['mean'], row['ci_low'], row['ci_high'], color,
                         whisker_h=_WHISKER_H_COMB_LP, whisker_lw=_WHISKER_LW_COMB_LP,
                         dot_size=_DOT_SIZE_COMB, bar_lw=_BAR_LW_COMB)

        ax.axvline(0, color='red', linestyle='--', linewidth=_ZERO_LW,
                   alpha=0.85, zorder=4)

        # Synchronise y-limits explicitly (replaces sharey)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(np.arange(n))
        if col_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=7.5)
        else:
            # tick_params hides marks+labels without touching the formatter
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.set_title(format_outcome(outcome), fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('ITE  (Bootstrap 95% CI)', fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.35, linewidth=0.8)
        _xlim_with_margin(ax, [df_s['ci_low'].values, df_s['ci_high'].values],
                          margin_frac=0.18)

    fig.suptitle(
        f"{method_name}  \u2014  ITE by Line\u2013Product\n"
        f"\u03c4 = SD of individual ITEs (within-group heterogeneity)  \u00b7  Bootstrap 95% CI",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout(pad=1.2, w_pad=0.5)
    h_in = fig.get_figheight()
    bottom_frac = max(0.03, 0.55 / h_in)
    fig.subplots_adjust(top=0.91, bottom=bottom_frac, right=0.98)
    return fig


# ----------------------------------------------------------------------------
# Plot 3 — ITE by Line (all outcomes side by side)
# ----------------------------------------------------------------------------

def plot_forest_plot_combined(meta_results_by_outcome, method_name=""):
    """
    Combined 3-panel figure: ITE by Line for all outcomes side by side.

    Y-axis labels (line names) appear on the leftmost panel only.
    I² values differ per outcome so are omitted from shared y-labels; the
    suptitle notes that I² = heterogeneity across products (Cochran Q).
    """
    outcomes = list(meta_results_by_outcome.keys())
    n_out    = len(outcomes)

    first_meta = meta_results_by_outcome[outcomes[0]]
    n_lines    = len(first_meta)

    # Simplified y-labels: line name only (no I² — it differs per outcome)
    y_labels = [f"Line {int(r['line_id'])}" for _, r in first_meta.iterrows()][::-1]

    fig_w = 6.5 * n_out
    fig_h = max(5.0, n_lines * 0.95 + 2.5)
    # sharey=False: avoid FixedFormatter-sharing bug (same reason as plot_ite_ci_combined).
    # Y-limits are synchronised manually below.
    fig, axes = plt.subplots(1, n_out, figsize=(fig_w, fig_h))
    if n_out == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    for col_idx, (outcome, ax) in enumerate(zip(outcomes, axes)):
        ax.set_facecolor('#EBEBEB')
        meta = meta_results_by_outcome[outcome]

        for idx, row in meta.iterrows():
            y     = n_lines - idx - 1
            color = LINE_COLORS.get(int(row['line_id']), 'gray')
            _draw_ci_row(ax, y, row['pooled_mean'], row['ci_low'], row['ci_high'], color,
                         whisker_h=_WHISKER_H_COMB_L, whisker_lw=_WHISKER_LW_COMB_L,
                         dot_size=_DOT_SIZE_COMB, bar_lw=_BAR_LW_COMB)

        ax.axvline(0, color='red', linestyle='--', linewidth=_ZERO_LW,
                   alpha=0.9, zorder=4)

        # Synchronise y-limits explicitly (replaces sharey)
        ax.set_ylim(-0.5, n_lines - 0.5)
        ax.set_yticks(np.arange(n_lines))
        if col_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=10)
        else:
            # tick_params hides marks+labels without touching the formatter
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.set_title(format_outcome(outcome), fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('')
        ax.grid(axis='x', alpha=0.4, linewidth=0.8)
        _xlim_with_margin(ax, [meta['ci_low'].values, meta['ci_high'].values],
                          margin_frac=0.22)

    fig.suptitle(
        f"{method_name}  \u2014  ITE by Line\n"
        f"I\u00b2 = heterogeneity across products (Cochran Q)  \u00b7  Bootstrap 95% CI",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout(pad=1.2, w_pad=0.5)
    h_in = fig.get_figheight()
    bottom_frac = max(0.04, 0.55 / h_in)
    fig.subplots_adjust(top=0.88, bottom=bottom_frac, right=0.98)
    return fig


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------------
    # 1. DATA LOADING
    # ------------------------------------------------------------------------
    df = load_data()
    outcomes = ['efficiency', 'wip', 'leadtime']

    # ------------------------------------------------------------------------
    # 2. MODEL SELECTION
    # ------------------------------------------------------------------------
    # Available meta-learners: 't-learner', 'x-learner', 'r-learner', 'dr-learner'
    # Available base models: 'linear', 'rf', 'gb'
    # Available tree methods: 'causal-forest', 'uplift-forest',
    #                         'interaction-tree', 'interaction-forest'

    single_method     = 'uplift-forest'
    single_base_model = 'gb'

    print(f"\n{'#' * 80}")
    print("# INDIVIDUAL MODEL ANALYSIS")
    print(f"{'#' * 80}")
    method_display_name = f"{single_method.upper()} ({single_base_model.upper()})"
    print(f"\nModel: {method_display_name}")
    print(f"Confidence: {CONFIDENCE_LEVEL*100:.0f}%  |  Bootstrap CI n={N_BOOTSTRAP:,}  |  Placebo permutations n={N_PLACEBO:,}")

    # ------------------------------------------------------------------------
    # 3. ITE ESTIMATION AND META-ANALYSIS
    # ------------------------------------------------------------------------
    all_results    = {}
    all_meta       = {}
    all_validation = {}

    for outcome in outcomes:
        print(f"\n{'-' * 80}")
        print(f"OUTCOME: {outcome.upper()}")
        print(f"{'-' * 80}")

        df_with_ite, learner = estimate_ite(df, outcome, single_method, single_base_model)

        # Placebo test (permutation CI + D statistic)
        validation_results = validate_model(df, outcome, single_method, single_base_model)
        all_validation[outcome] = validation_results

        # ITE aggregation: bootstrap CI per (line, product) + tau heterogeneity
        agg_results = aggregate_by_line_product(df_with_ite, outcome)
        all_results[outcome] = (df_with_ite, agg_results)

        # Meta-analysis: bootstrap CI per line + I² heterogeneity
        meta_results = meta_analysis_by_line(agg_results)
        all_meta[outcome] = meta_results

        print(f"\nITE BY LINE-PRODUCT  (Bootstrap 95% CI | τ = ITE heterogeneity within group):")
        print(tabulate(
            agg_results[['line_product', 'n', 'mean', 'ci_low', 'ci_high', 'tau']],
            headers=['Line-Product', 'n', 'Mean ITE', 'CI Lower', 'CI Upper', 'τ (SD)'],
            tablefmt='grid', floatfmt='.4f', showindex=False
        ))

        print(f"\nMETA-ANALYSIS BY LINE  (Bootstrap 95% CI | I² = heterogeneity across products):")
        print(tabulate(
            meta_results[['line_id', 'n_products', 'pooled_mean', 'ci_low', 'ci_high', 'I2', 'Q']],
            headers=['Line', 'N Products', 'Pooled ITE', 'CI Lower', 'CI Upper', 'I² (%)', 'Q'],
            tablefmt='grid', floatfmt='.4f', showindex=False
        ))

    # ------------------------------------------------------------------------
    # 4. PLACEBO TEST TABLE
    # ------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"PLACEBO TEST VALIDATION — PERMUTATION (n={N_PLACEBO:,})")
    print(f"{'=' * 80}")
    print("Method         : fit model on randomized treatment, repeat N_PLACEBO times")
    print("CI             : 2.5%–97.5% percentile of permutation mean-ITE distribution")
    print("Pass criterion : permutation CI contains zero")
    print("D statistic    : |median| / CI_width  (lower D → centred closer to zero)")
    print("Model selection: among passing models, prefer smallest max(D)")
    print(f"{'─' * 80}")

    placebo_tbl = []
    for outcome, res in all_validation.items():
        placebo_tbl.append({
            'Outcome':  format_outcome(outcome),
            'Mean':     f"{res['placebo_mean']:.4f}",
            'CI Lower': f"{res['ci_low']:.4f}",
            'CI Upper': f"{res['ci_high']:.4f}",
            'Valid?':   'PASS' if res['is_valid'] else 'FAIL',
            'D stat':   f"{res['d_stat']:.4f}",
        })
    print(tabulate(placebo_tbl, headers='keys', tablefmt='grid'))

    all_valid = all(all_validation[o]['is_valid'] for o in outcomes)
    max_d     = max(all_validation[o]['d_stat'] for o in outcomes)
    print(f"\nModel: {method_display_name}")
    print(f"  Passes ALL placebo tests : {'YES' if all_valid else 'NO'}")
    print(f"  Max D across outcomes    : {max_d:.4f}")
    print("  (When comparing models: keep those that pass all; prefer smallest max D)")
    print(f"{'─' * 80}")

    # ------------------------------------------------------------------------
    # 5. VISUALIZATION
    # ------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("GENERATING PLOTS...")
    print(f"{'=' * 80}")

    # --- Combined ITE by Line-Product (all 3 outcomes side by side) ---
    print("\nCombined ITE by Line-Product plot (all outcomes side by side)...")
    results_by_outcome = {o: all_results[o][1] for o in outcomes}
    fig1 = plot_ite_ci_combined(results_by_outcome, method_display_name)
    plt.show(block=False)

    # --- Combined ITE by Line (all 3 outcomes side by side) ---
    print("Combined ITE by Line plot (all outcomes side by side)...")
    fig2 = plot_forest_plot_combined(all_meta, method_display_name)
    plt.show(block=False)

    # --- Placebo test: single combined plot for all outcomes ---
    print("Placebo test plot (all outcomes combined)...")
    fig3 = plot_validation(all_validation, method_display_name)
    plt.show(block=False)

    plt.show()

    # ------------------------------------------------------------------------
    # 6. COMPLETION
    # ------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("# COMPLETED!")
    print(f"{'#' * 80}")

if __name__ == "__main__":
    main()
