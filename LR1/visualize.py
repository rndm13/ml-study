import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(
    data=df,
    x='sepal width (cm)',
    kde=True,
    color='teal',
    ax=axes[0, 0]
)
axes[0, 0].set_title('1. Розподіл цільової змінної (Sepal Width)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Ширина чашолистка (см)')
axes[0, 0].set_ylabel('Частота')

sns.boxplot(
    data=df,
    x='species',
    y='sepal width (cm)',
    hue='species',
    palette='Set2',
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title('2. Розподіл Sepal Width за видами ірисів', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Вид ірису')
axes[0, 1].set_ylabel('Ширина чашолистка (см)')

sns.scatterplot(
    data=df,
    x='petal length (cm)',
    y='sepal width (cm)',
    hue='species',
    palette='Set2',
    s=70,
    ax=axes[1, 0]
)
axes[1, 0].set_title('3. Взаємозв\'язок: Petal Length vs Sepal Width', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Довжина пелюстки (см)')
axes[1, 0].set_ylabel('Ширина чашолистка (см)')

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
