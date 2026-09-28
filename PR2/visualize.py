import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load data
iris = load_iris(as_frame=True)
df = iris.frame

print(f"Size of dataset: {df.shape}")
print(f"Skipped rows: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print(f"{df.describe()}")

df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

target_col = 'sepal width (cm)'

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1

L = Q1 - 1.5 * IQR  # Bottom threshold
U = Q3 + 1.5 * IQR  # Top threshold

df['is_outlier'] = (df[target_col] < L) | (df[target_col] > U)
df['is_outlier_setosa'] = (df['species'] == 'setosa') & (df[target_col] < L) | (df[target_col] > U)

outliers = df[df['is_outlier']]
outliers_setosa = df[df['is_outlier_setosa']]
print(f"--- Статистика виявлення аномалій (IQR) ---")
print(f"Q1 (25%): {Q1:.2f} см | Q3 (75%): {Q3:.2f} см | IQR: {IQR:.2f} см")
print(f"Нижня межа (L): {L:.2f} см | Верхня межа (U): {U:.2f} см")
print(f"Знайдено аномальних об'єктів: {len(outliers)}")
if not outliers.empty:
    print(outliers[['sepal length (cm)', 'sepal width (cm)', 'species']])

y = df['sepal width (cm)']
X = df.drop(columns=['sepal width (cm)', 'target'])

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with IQR
sns.histplot(
    data=df,
    x=target_col,
    kde=True,
    color='teal',
    ax=axes[0, 0]
)
axes[0, 0].axvline(L, color='red', linestyle='--', linewidth=1.5, label=f'L = {L:.2f}')
axes[0, 0].axvline(U, color='red', linestyle='--', linewidth=1.5, label=f'U = {U:.2f}')
axes[0, 0].set_title('1. Розподіл Sepal Width та межі IQR (L, U)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Ширина чашолистка (см)')
axes[0, 0].legend()

# Boxplot by species
sns.boxplot(
    data=df,
    x='species',
    y=target_col,
    hue='species',
    palette='Set2',
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].axhline(L, color='red', linestyle='--', alpha=0.7)
axes[0, 1].axhline(U, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('2. Boxplot із межами розмаху (L, U)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Вид ірису')

# Graph 3
sns.scatterplot(
    data=df,
    x='petal length (cm)',
    y=target_col,
    hue='species',
    palette='Set2',
    s=70,
    ax=axes[1, 0]
)
axes[1, 0].set_title('3. Взаємозв\'язок: Petal Length vs Sepal Width', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Довжина пелюстки (см)')
axes[1, 0].set_ylabel('Ширина чашолистка (см)')

# Graph 4
numeric_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    ax=axes[1, 1]
)
axes[1, 1].set_title('4. Теплова карта кореляцій (Heatmap)', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.savefig('iris_data_visualization.png', dpi=300)
print("Графік успішно збережено у файл 'iris_data_visualization.png'")

plt.show()
