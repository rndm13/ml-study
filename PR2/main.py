import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD
iris = load_iris(as_frame=True)
df = iris.frame

y = df['sepal width (cm)']
X = df.drop(columns=['sepal width (cm)', 'target'])
X['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

num_features = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
cat_features = ['species']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# TRAIN TEST SPLIT 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor.fit(X_train)
preprocessor.transform(X_test)

# CANDIDATE MODELS
baseline = DummyRegressor(strategy="mean")

# Candidate 1: Linear Regression
linear_model = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Candidate 2: Polynomial Regression (degree=2)
poly2_model = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

# Candidate 3: Polynomial Regression (degree=3)
poly3_model = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("model", LinearRegression())
])

# Candidate 4: Ridge Regression
ridge_model = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# Candidate 5: Decision tree
decision_tree_model = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler()),
    ("model", DecisionTreeRegressor(random_state=42)),
])

models = {
    "Baseline": baseline,
    "Linear": linear_model,
    "Polynomial (d=2)": poly2_model,
    "Polynomial (d=3)": poly3_model,
    "Ridge": ridge_model,
    "Decision tree": decision_tree_model,
}

# K-FOLD CROSS-VALIDATION (k = 5)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -scores
    cv_results.append({
        "Model": name,
        "CV_RMSE_mean": rmse_scores.mean(),
        "CV_RMSE_std": rmse_scores.std()
    })

results_df = pd.DataFrame(cv_results)
print("\n--- Cross-validation results (Train) ---")
print(results_df.to_string(index=False))

# FINAL TRAINING

final_model = ridge_model
final_model.fit(X_train, y_train)
baseline.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred_base = baseline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Final model evaluation ---")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

# EVALUATION GRAPHS

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 7))

plt.bar(
    results_df["Model"],
    results_df["CV_RMSE_mean"],
    yerr=results_df["CV_RMSE_std"],
    capsize=5
)

plt.ylabel("RMSE"),
plt.title("5-fold cross-validation")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig('5_fold_CV.png', dpi=300)
print("Saved 5-fold cross-validation to '5_fold_CV.png'")
plt.show()

plt.scatter(
    y_test, y_pred,
    color='darkblue', alpha=0.75, s=70,
    edgecolor='k', label='Final model predictions'
)

plt.axhline(
    y=y_train.mean(), color='crimson', linestyle=':',
    linewidth=2, label=f'Baseline (Mean = {y_train.mean():.2f} cm)'
)

min_val = min(y_test.min(), y_pred.min()) - 0.2
max_val = max(y_test.max(), y_pred.max()) + 0.2
plt.plot(
    [min_val, max_val], [min_val, max_val],
    color='red', linestyle='--', linewidth=2,
    label='Ідеальне передбачення ($y_{pred} = y_{true}$)'
)

plt.title('Порівняння передбачень: $y_{true}$ проти $y_{predicted}$ (Test Set)', fontsize=13, fontweight='bold')
plt.xlabel('Фактичні значення $y_{true}$ (см)', fontsize=11)
plt.ylabel('Передбачені значення $y_{predicted}$ (см)', fontsize=11)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('y_true_vs_y_pred.png', dpi=300)
print("Final analysis graph saved to 'y_true_vs_y_pred.png'")
plt.show()

# ERROR ANALYSIS
residuals = y_test - y_pred
errors_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals,
    'Abs_Error': np.abs(residuals)
})

largest_errors = errors_df.sort_values(by='Abs_Error', ascending=False).head(5)
print("\n--- 5 worst predictions ---")
print(largest_errors.to_string())

# RESIDUAL GRAPH
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals,
    color='crimson',
    s=70,
    alpha=0.8,
    ax=axes[0]
)
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)

sns.regplot(
    x=y_pred,
    y=residuals,
    scatter=False,
    color='blue',
    ax=axes[0],
    line_kws={'linestyle': ':', 'linewidth': 2}
)

axes[0].set_title('Графік залишків: $\hat{y}$ проти $e_i$', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Передбачені значення $\hat{y}$ (см)')
axes[0].set_ylabel('Залишки $e_i = y_i - \hat{y}_i$ (см)')

sns.histplot(
    residuals,
    kde=True,
    color='teal',
    ax=axes[1]
)
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_title('Розподіл залишків $e_i$', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Залишок $e_i$ (см)')
axes[1].set_ylabel('Частота')
plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300)
print("Graph of residual analysis saved to 'residuals_analysis.png'")
plt.show()
