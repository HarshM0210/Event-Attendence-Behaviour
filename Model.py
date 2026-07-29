"""
Event Attendance Prediction
================================================ 
Target   : % Tickets Sold (continuous, 7.7–100.0)

FEATURES (9 total):
  Base (5):
    - Month_sin          : sin(2*pi*month/12) cyclical encoding
    - Month_cos          : cos(2*pi*month/12) cyclical encoding
    - EventCat_encoded   : target encoding (smoothing=10, train set only)
    - Location_encoded   : target encoding (smoothing=10, train set only)
    - Event_Scaled       : Small=0, Medium=1, Large=2

  Interaction (4):
    - Weekend_x_Holiday  : Is_Weekend_Day x Next_Day_Holiday
    - Cat_x_Holiday      : EventCat_encoded x Next_Day_Holiday
    - Cat_x_Weekend      : EventCat_encoded x Is_Weekend_Day
    - Scale_x_Cat        : Event_Scaled x EventCat_encoded

MODELS:
  1. XGBoost  — custom squared epsilon loss: max(0, (|e| - 10)^2)
  2. Random Forest — sample weighted (0-50%: 2x, 85-100%: 4x)

SPLIT  : 80/20 stratified on quartile bins of target, random_state=42
"""

import numpy as np
import pandas as pd
import warnings, math
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import openpyxl

WINDOW = 20


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data(path='Events.xlsx', sheet='Final'):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    headers   = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column+1)]
    col       = {h: i+1 for i, h in enumerate(headers)}
    real_rows = [r for r in range(2, ws.max_row+1)
                 if ws.cell(row=r, column=1).value is not None]

    MONTH_NUM = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                 'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    SCALE_ENC = {'Small':0,'Medium':1,'Large':2}

    data = []
    for r in real_rows:
        day = ws.cell(row=r, column=col['Day']).value
        m   = MONTH_NUM[ws.cell(row=r, column=col['Month']).value]
        data.append({
            'Month_sin'       : round(math.sin(2*math.pi*m/12), 6),
            'Month_cos'       : round(math.cos(2*math.pi*m/12), 6),
            'Location_raw'    : ws.cell(row=r, column=col['Location']).value,
            'Event_Scaled'    : SCALE_ENC[ws.cell(row=r, column=col['Event Scale']).value],
            'EventCat_raw'    : ws.cell(row=r, column=col['Event Category']).value,
            'Is_Weekend_Day'  : 1 if day in ('Friday','Saturday','Sunday') else 0,
            'Next_Day_Holiday': int(float(ws.cell(row=r, column=col['Next day Holiday?']).value)),
            'Tickets_Sold'    : float(ws.cell(row=r, column=col['% Tickets Sold']).value),
        })

    df = pd.DataFrame(data).drop_duplicates()
    print(f"Rows loaded  : {len(real_rows)}")
    print(f"After dedup  : {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — SPLIT + ENCODE + BUILD FEATURES
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'Month_sin', 'Month_cos',
    'EventCat_encoded', 'Location_encoded', 'Event_Scaled',
    'Weekend_x_Holiday', 'Cat_x_Holiday', 'Cat_x_Weekend', 'Scale_x_Cat',
]

def split_and_encode(df, test_size=0.2, random_state=42, smoothing=10):
    X_raw = df.drop(columns=['Tickets_Sold'])
    y     = df['Tickets_Sold']

    y_binned = pd.qcut(y, q=4, labels=False, duplicates='drop')
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state,
        stratify=y_binned)

    global_mean = y_train.mean()

    def target_encode(tr_col, te_col):
        stats      = y_train.groupby(tr_col).agg(['mean', 'count'])
        smooth_enc = (stats['count']*stats['mean'] + smoothing*global_mean) / \
                     (stats['count'] + smoothing)
        return (tr_col.map(smooth_enc).fillna(global_mean).round(4),
                te_col.map(smooth_enc).fillna(global_mean).round(4))

    cat_tr, cat_te = target_encode(X_train_raw['EventCat_raw'], X_test_raw['EventCat_raw'])
    loc_tr, loc_te = target_encode(X_train_raw['Location_raw'], X_test_raw['Location_raw'])

    def build_X(Xr, ct, lt):
        out = Xr.copy()
        out['EventCat_encoded']  = ct.values
        out['Location_encoded']  = lt.values
        out['Weekend_x_Holiday'] = out['Is_Weekend_Day']   * out['Next_Day_Holiday']
        out['Cat_x_Holiday']     = out['EventCat_encoded'] * out['Next_Day_Holiday']
        out['Cat_x_Weekend']     = out['EventCat_encoded'] * out['Is_Weekend_Day']
        out['Scale_x_Cat']       = out['Event_Scaled']     * out['EventCat_encoded']
        return out[FEATURE_COLS]

    X_train = build_X(X_train_raw, cat_tr, loc_tr)
    X_test  = build_X(X_test_raw,  cat_te, loc_te)

    print(f"\nTrain rows   : {len(X_train)}")
    print(f"Test  rows   : {len(X_test)}")
    print(f"Features     : {len(FEATURE_COLS)}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
# STEP 3 — CUSTOM LOSS (XGBoost)
# L = max(0, (|actual - predicted| - 10)^2)
# ─────────────────────────────────────────────────────────────────

def squared_epsilon_loss(y_true, y_pred, epsilon=10):
    residual = y_true - y_pred
    abs_e    = np.abs(residual)
    grad = np.where(
        abs_e <= epsilon, 0.0,
        np.where(residual > 0,
                 -2*(residual - epsilon),
                 -2*(residual + epsilon))
    )
    hess = np.where(abs_e <= epsilon, 1e-6, 2.0)
    return grad, hess


# ─────────────────────────────────────────────────────────────────
# STEP 4 — METRIC
# ─────────────────────────────────────────────────────────────────

def compute_accuracy(y_actual, y_pred):
    hits = (np.abs(np.array(y_pred) - np.array(y_actual)) <= WINDOW).astype(int)
    return hits, hits.mean() * 100

def print_results(name, tr_hits, tr_acc, te_hits, te_acc, y_train, y_test):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Train accuracy : {tr_acc:.2f}%  ({tr_hits.sum()} / {len(tr_hits)})")
    print(f"  Test  accuracy : {te_acc:.2f}%  ({te_hits.sum()} / {len(te_hits)})")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — MODELS
# ─────────────────────────────────────────────────────────────────

def run_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        objective        = squared_epsilon_loss,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
        n_estimators     = 500,
        learning_rate    = 0.04,
        max_depth        = 5,
        min_child_weight = 3,
        reg_alpha        = 0.3,
        reg_lambda       = 1.0,
        subsample        = 0.80,
        colsample_bytree = 0.80,
        gamma            = 0.05,
    )
    model.fit(X_train, y_train)
    tr_hits, tr_acc = compute_accuracy(y_train, model.predict(X_train))
    te_hits, te_acc = compute_accuracy(y_test,  model.predict(X_test))
    print_results("XGBoost — Squared Epsilon Loss",
                  tr_hits, tr_acc, te_hits, te_acc, y_train, y_test)

    return model, tr_acc, te_acc


def run_random_forest(X_train, X_test, y_train, y_test):
    # Sample weights: upweight extreme attendance ranges
    sample_weights = np.ones(len(y_train))
    sample_weights[np.array(y_train) < 50]  = 2.0   # low attendance: 2x
    sample_weights[np.array(y_train) >= 85] = 4.0   # high attendance: 4x

    model = RandomForestRegressor(
        n_estimators     = 300,
        max_depth        = None,
        min_samples_leaf = 4,
        max_features     = 'sqrt',
        bootstrap        = True,
        max_samples      = 0.85,
        random_state     = 42,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    tr_hits, tr_acc = compute_accuracy(y_train, model.predict(X_train))
    te_hits, te_acc = compute_accuracy(y_test,  model.predict(X_test))
    print_results("Random Forest — Sample Weighted (0-50%: 2x, 85-100%: 4x)",
                  tr_hits, tr_acc, te_hits, te_acc, y_train, y_test)
    return model, tr_acc, te_acc


# ─────────────────────────────────────────────────────────────────
# STEP 6 — COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────

def print_comparison(xgb_tr, xgb_te, rf_tr, rf_te):
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — ±10pp Accuracy")
    print(f"  9 features  |  80/20 stratified split")
    print(f"{'='*60}")
    print(f"  {'Model':<38} {'Train Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*38} {'-'*10} {'-'*10}")
    print(f"  {'XGBoost (Sq-epsilon loss)':<38} {xgb_tr:>9.2f}% {xgb_te:>9.2f}%")
    print(f"  {'RF (sample weighted)':<38} {rf_tr:>9.2f}% {rf_te:>9.2f}%")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print(f"Metric: ±10pp accuracy\n")

    df = load_data(path='Events.xlsx', sheet='Final')
    X_train, X_test, y_train, y_test = split_and_encode(df)

    xgb_model, xgb_tr, xgb_te = run_xgboost(X_train, X_test, y_train, y_test)
    rf_model,  rf_tr,  rf_te  = run_random_forest(X_train, X_test, y_train, y_test)

    print_comparison(xgb_tr, xgb_te, rf_tr, rf_te)