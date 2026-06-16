import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
data = pd.read_excel("salary_data.xlsx")

print("── Raw Data Preview ──")
print(data.head(10))
print(data.info())
print(data.describe())

# Rename columns to standard names if needed — adjust to match your actual column headers
data.columns = data.columns.str.strip()
data = data.rename(columns={
    "Location Code":        "location_code",
    "Job Code":             "job_code",
    "Cost Center ID":       "cost_center_id",
    "Compensation Grade":   "comp_grade",
    "Total Base Pay (USD)": "base_pay",
    "Job Req Filled Date":  "filled_date"
})

# ─────────────────────────────────────────────
# 2. PARSE DATE & EXTRACT YEAR
# ─────────────────────────────────────────────
data["filled_date"] = pd.to_datetime(data["filled_date"])
data["year"] = data["filled_date"].dt.year

print("\nYears present in dataset:", sorted(data["year"].unique()))

# ─────────────────────────────────────────────
# 3. DATA VALIDATION
# ─────────────────────────────────────────────
print("\n── Missing Values (%) ──")
missing = data.isna().sum() * 100 / len(data)
print(missing[missing > 0])

# Drop rows missing the target or key identifiers
data = data.dropna(subset=["base_pay", "location_code", "job_code",
                            "cost_center_id", "comp_grade", "year"])

# Remove zero or negative pay
invalid_pay = (data["base_pay"] <= 0).sum()
print(f"\nRows with zero/negative base pay removed: {invalid_pay}")
data = data[data["base_pay"] > 0]

print(f"\nClean dataset shape: {data.shape}")
print("\nRecords per year:")
print(data["year"].value_counts().sort_index())

print("\nRecords per compensation grade:")
print(data["comp_grade"].value_counts().sort_index())

# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

# Distribution of base pay (before log)
plt.figure(figsize=(10, 5))
sns.histplot(data["base_pay"], bins=40, kde=True)
plt.title("Distribution of Base Pay (Before Log Transform)")
plt.xlabel("Base Pay (USD)")
plt.tight_layout()
plt.savefig("eda_base_pay_distribution.png", dpi=150)
plt.show()

# Log-transformed distribution
plt.figure(figsize=(10, 5))
sns.histplot(np.log(data["base_pay"]), bins=40, kde=True, color="steelblue")
plt.title("Distribution of Log(Base Pay)")
plt.xlabel("Log Base Pay")
plt.tight_layout()
plt.savefig("eda_log_base_pay_distribution.png", dpi=150)
plt.show()

# Median base pay by compensation grade
plt.figure(figsize=(10, 5))
grade_pay = data.groupby("comp_grade")["base_pay"].median().sort_values()
grade_pay.plot(kind="bar", color="steelblue")
plt.title("Median Base Pay by Compensation Grade")
plt.xlabel("Compensation Grade")
plt.ylabel("Median Base Pay (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_pay_by_grade.png", dpi=150)
plt.show()

# Median base pay by year
plt.figure(figsize=(8, 4))
year_pay = data.groupby("year")["base_pay"].median()
year_pay.plot(kind="line", marker="o", color="steelblue")
plt.title("Median Base Pay by Year")
plt.xlabel("Year")
plt.ylabel("Median Base Pay (USD)")
plt.tight_layout()
plt.savefig("eda_pay_by_year.png", dpi=150)
plt.show()

# Boxplot by year
plt.figure(figsize=(10, 5))
sns.boxplot(data=data, x="year", y="base_pay")
plt.title("Base Pay Distribution by Year")
plt.xlabel("Year")
plt.ylabel("Base Pay (USD)")
plt.tight_layout()
plt.savefig("eda_boxplot_by_year.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# 5a. Ordinal encoding for compensation grade
#     List grades in ascending order — adjust this list to match your actual grade values
grade_order = sorted(data["comp_grade"].unique().tolist())
print("\nCompensation grade order (adjust if needed):", grade_order)

oe = OrdinalEncoder(categories=[grade_order])
data["comp_grade_enc"] = oe.fit_transform(data[["comp_grade"]])

# 5b. Label encoding for cost center ID (many categories, not ordinal)
le_cc = LabelEncoder()
data["cost_center_enc"] = le_cc.fit_transform(data["cost_center_id"].astype(str))

# 5c. Label encoding for location code and job code
le_loc = LabelEncoder()
data["location_enc"] = le_loc.fit_transform(data["location_code"].astype(str))

le_job = LabelEncoder()
data["job_enc"] = le_job.fit_transform(data["job_code"].astype(str))

# 5d. Aggregate features — median pay by key groupings
#     These give the model the pay signal for each combination
data["median_pay_job_grade"] = data.groupby(
    ["job_code", "comp_grade"])["base_pay"].transform("median")

data["median_pay_loc_grade"] = data.groupby(
    ["location_code", "comp_grade"])["base_pay"].transform("median")

data["median_pay_cc_grade"] = data.groupby(
    ["cost_center_id", "comp_grade"])["base_pay"].transform("median")

# 5e. Log-transform the target
data["log_base_pay"] = np.log(data["base_pay"])

print("\nFeature engineering complete. Sample:")
print(data[["job_code", "comp_grade", "comp_grade_enc",
            "cost_center_enc", "location_enc",
            "median_pay_job_grade", "log_base_pay"]].head())

# ─────────────────────────────────────────────
# 6. TIME-BASED TRAIN / VALIDATE / TEST SPLIT
#    Train: 2021–2024 | Validate: 2025 | Test: 2026
# ─────────────────────────────────────────────
FEATURES = [
    "job_enc",
    "location_enc",
    "cost_center_enc",
    "comp_grade_enc",
    "year",
    "median_pay_job_grade",
    "median_pay_loc_grade",
    "median_pay_cc_grade"
]
TARGET = "log_base_pay"

train = data[data["year"] <= 2024]
validate = data[data["year"] == 2025]
test = data[data["year"] == 2026]

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validate[FEATURES]
y_val = validate[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print(f"\nTrain size:    {len(X_train)}")
print(f"Validate size: {len(X_val)}")
print(f"Test size:     {len(X_test)}")


# ─────────────────────────────────────────────
# 7. METRIC HELPER
# ─────────────────────────────────────────────
def evaluate(y_true_log, y_pred_log, model_name="Model"):
    """
    y_true_log and y_pred_log are in log space.
    Exponentiate back to USD before computing MAPE.
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true_log, y_pred_log)   # R² in log space

    print(f"\n── {model_name} ──")
    print(f"  MAPE : {mape:.2f}%  {'✓ Target met' if mape <= 5 else '✗ Above 5% target'}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

    return {"Model": model_name, "MAPE (%)": round(mape, 2),
            "MAE ($)": round(mae, 0), "RMSE ($)": round(rmse, 0),
            "R²": round(r2, 4)}


# ─────────────────────────────────────────────
# 8. MODEL 1 — DECISION TREE (baseline)
# ─────────────────────────────────────────────
DT = DecisionTreeRegressor(max_depth=5, random_state=42)
DT.fit(X_train, y_train)
y_pred_dt = DT.predict(X_val)
results_dt = evaluate(y_val, y_pred_dt, "Decision Tree (baseline)")

# ─────────────────────────────────────────────
# 9. MODEL 2 — RANDOM FOREST
# ─────────────────────────────────────────────
RF = RandomForestRegressor(n_estimators=200, max_depth=None,
                           min_samples_leaf=5, random_state=42, n_jobs=-1)
RF.fit(X_train, y_train)
y_pred_rf = RF.predict(X_val)
results_rf = evaluate(y_val, y_pred_rf, "Random Forest")

# ─────────────────────────────────────────────
# 10. MODEL 3 — SKLEARN GRADIENT BOOSTING
# ─────────────────────────────────────────────
GB = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300,
                               max_depth=4, min_samples_leaf=5,
                               subsample=0.8, random_state=42)
GB.fit(X_train, y_train)
y_pred_gb = GB.predict(X_val)
results_gb = evaluate(y_val, y_pred_gb, "Gradient Boosting (sklearn)")

# ─────────────────────────────────────────────
# 11. MODEL 4 — XGBOOST
# ─────────────────────────────────────────────
XGB = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
XGB.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False)
y_pred_xgb = XGB.predict(X_val)
results_xgb = evaluate(y_val, y_pred_xgb, "XGBoost")

# ─────────────────────────────────────────────
# 12. MODEL 5 — LIGHTGBM
# ─────────────────────────────────────────────
LGB = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
LGB.fit(X_train, y_train,
        eval_set=[(X_val, y_val)])
y_pred_lgb = LGB.predict(X_val)
results_lgb = evaluate(y_val, y_pred_lgb, "LightGBM")

# ─────────────────────────────────────────────
# 13. RESULTS COMPARISON TABLE
# ─────────────────────────────────────────────
results_df = pd.DataFrame([
    results_dt, results_rf, results_gb, results_xgb, results_lgb
]).set_index("Model")

print("\n════════════════════════════════════════")
print("       MODEL COMPARISON (Validate 2025)")
print("════════════════════════════════════════")
print(results_df.to_string())
print("════════════════════════════════════════")

best_model_name = results_df["MAPE (%)"].idxmin()
print(f"\nBest model by MAPE: {best_model_name}")

# ─────────────────────────────────────────────
# 14. BAR CHART — MAPE COMPARISON
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 5))
colors = ["#e07b54" if m != best_model_name else "#2a7ec8"
          for m in results_df.index]
bars = plt.bar(results_df.index, results_df["MAPE (%)"], color=colors)
plt.axhline(5, color="red", linestyle="--", linewidth=1, label="5% MAPE target")
plt.title("MAPE by Model — Validation Set (2025)", fontsize=14)
plt.xlabel("Model")
plt.ylabel("MAPE (%)")
plt.xticks(rotation=15, ha="right")
plt.legend()
for bar, val in zip(bars, results_df["MAPE (%)"]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1, f"{val:.2f}%",
             ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("model_mape_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 15. FEATURE IMPORTANCE — BEST MODEL
# ─────────────────────────────────────────────
model_map = {
    "Decision Tree (baseline)": DT,
    "Random Forest": RF,
    "Gradient Boosting (sklearn)": GB,
    "XGBoost": XGB,
    "LightGBM": LGB
}
best_model = model_map[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="steelblue")
    plt.title(f"Feature Importance — {best_model_name}", fontsize=13)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# 16. ACTUAL vs PREDICTED SCATTER — BEST MODEL
# ─────────────────────────────────────────────
if best_model_name == "XGBoost":
    y_pred_best = y_pred_xgb
elif best_model_name == "LightGBM":
    y_pred_best = y_pred_lgb
elif best_model_name == "Random Forest":
    y_pred_best = y_pred_rf
elif best_model_name == "Gradient Boosting (sklearn)":
    y_pred_best = y_pred_gb
else:
    y_pred_best = y_pred_dt

actual_pay = np.exp(y_val.values)
pred_pay   = np.exp(y_pred_best)

plt.figure(figsize=(7, 7))
plt.scatter(actual_pay, pred_pay, alpha=0.4, s=20, color="steelblue")
min_val = min(actual_pay.min(), pred_pay.min())
max_val = max(actual_pay.max(), pred_pay.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1,
         label="Perfect prediction")
plt.title(f"Actual vs Predicted Base Pay — {best_model_name}", fontsize=13)
plt.xlabel("Actual Base Pay (USD)")
plt.ylabel("Predicted Base Pay (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 17. HOLDOUT TEST — 2026 YTD (if data exists)
# ─────────────────────────────────────────────
if len(X_test) > 0:
    print("\n── Holdout Test: 2026 YTD ──")
    if best_model_name == "XGBoost":
        y_pred_test = XGB.predict(X_test)
    elif best_model_name == "LightGBM":
        y_pred_test = LGB.predict(X_test)
    elif best_model_name == "Random Forest":
        y_pred_test = RF.predict(X_test)
    elif best_model_name == "Gradient Boosting (sklearn)":
        y_pred_test = GB.predict(X_test)
    else:
        y_pred_test = DT.predict(X_test)

    holdout_results = evaluate(y_test, y_pred_test,
                               f"{best_model_name} — 2026 Holdout")
else:
    print("\nNo 2026 data available yet for holdout test.")

# ─────────────────────────────────────────────
# 18. OPTIONAL HYPERPARAMETER TUNING (XGBoost)
#     Uncomment and run after initial evaluation
# ─────────────────────────────────────────────
# param_grid = {
#     "max_depth":      [3, 5, 7],
#     "learning_rate":  [0.01, 0.05, 0.1],
#     "n_estimators":   [200, 400, 600],
#     "subsample":      [0.7, 0.8, 1.0],
#     "colsample_bytree": [0.7, 0.8, 1.0]
# }
# xgb_tuned = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
# gs = GridSearchCV(xgb_tuned, param_grid, cv=3,
#                   scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
# gs.fit(X_train, y_train)
# print("Best XGBoost params:", gs.best_params_)

print("\nDone. All plots saved to working directory.")
