import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Ustawienia dla lepszej wizualizacji
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def load_california_housing():
    # Pobranie danych
    housing = fetch_california_housing()
    
    # Konwersja na DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MEDV'] = housing.target  # Dodanie zmiennej celu (mediany wartości domów)
    
    print(f"Załadowano zestaw danych California Housing: {df.shape[0]} obserwacji, {df.shape[1]} zmiennych")
    print("\nOpis zmiennych:")
    for i, feature in enumerate(housing.feature_names):
        print(f"  - {feature}: {housing.feature_names[i]}")
    print(f"  - MEDV: Zmienna celu - mediana wartości domów w tysiącach dolarów")
    
    return df, housing.feature_names

def show_basic_stats(df):
    print("\nPodstawowe statystyki:")
    print(df.describe().round(2))
    
    # Wyliczenie korelacji z celem
    correlations = df.corr()['MEDV'].sort_values(ascending=False)
    print("\nKorelacja zmiennych z ceną domów (MEDV):")
    print(correlations)
    
    return correlations

def plot_distribution_of_features(df, feature_names):
    """Wizualizacja rozkładu wszystkich cech"""
    n_features = len(feature_names)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Histogramy dla wszystkich cech
    for i, feature in enumerate(feature_names):
        ax = plt.subplot(gs[i])
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue', alpha=0.8)
        ax.set_title(f'Rozkład: {feature}', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Liczebność')
        
    # Histogram dla zmiennej celu (MEDV)
    ax = plt.subplot(gs[n_features])
    sns.histplot(df['MEDV'], kde=True, ax=ax, color='crimson', alpha=0.8)
    ax.set_title('Rozkład: Mediana Wartości Domów (MEDV)', fontsize=12)
    ax.set_xlabel('Tysiące dolarów')
    ax.set_ylabel('Liczebność')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df):
    """Wizualizacja macierzy korelacji"""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title('Macierz korelacji cech', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pairplot(df, feature_names):
    """Wizualizacja pairplota dla najważniejszych zmiennych"""
    # Wybieramy 4 najważniejsze cechy + cel
    top_features = df[list(feature_names[:4]) + ['MEDV']]
    
    g = sns.pairplot(top_features, diag_kind='kde', 
                     plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                     height=2.5, corner=True)
    
    g.fig.suptitle('Wykres par dla najważniejszych zmiennych', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_visualization(df, feature_names):
    """Wizualizacja 3D dla 2 najważniejszych zmiennych i celu"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Wybieramy 2 najważniejsze cechy
    x = df[feature_names[0]]
    y = df[feature_names[1]]
    z = df['MEDV']
    
    # Stosujemy kolorowanie wg mediany wartości domów
    p = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.7, edgecolor='w')
    
    ax.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax.set_zlabel('Mediana Wartości Domów (MEDV)', fontsize=12, labelpad=10)
    
    ax.view_init(elev=30, azim=45)
    
    # Dodanie paska kolorów
    cb = fig.colorbar(p, ax=ax, pad=0.1)
    cb.set_label('Mediana Wartości Domów (w tys. $)', fontsize=12)
    
    plt.title('Wizualizacja 3D zależności cen domów od dwóch głównych cech', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(correlations):
    """Wizualizacja ważności cech na podstawie korelacji z celem"""
    # Usuwamy MEDV z korelacji
    correlations = correlations.drop('MEDV')
    
    # Sortujemy według wartości bezwzględnej korelacji
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations[correlations_abs.index]]
    
    ax = sns.barplot(x=correlations_abs.values, y=correlations_abs.index, palette=colors)
    
    # Dodanie wartości korelacji na końcach słupków
    for i, v in enumerate(correlations[correlations_abs.index].values):
        ax.text(0.01, i, f' {v:.2f}', color='black', va='center', fontweight='bold')
    
    plt.title('Ważność cech (korelacja z ceną domów)', fontsize=14)
    plt.xlabel('Wartość bezwzględna współczynnika korelacji')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Wczytanie danych
    df, feature_names = load_california_housing()
    
    # Wyświetlenie podstawowych statystyk
    correlations = show_basic_stats(df)
    
    print("\nTworzenie wizualizacji danych...")
    
    # Wizualizacje
    plot_distribution_of_features(df, feature_names)
    plot_correlation_heatmap(df)
    plot_pairplot(df, feature_names)
    plot_3d_visualization(df, feature_names)
    plot_feature_importance(correlations)
    
    print("\nWizualizacja danych California Housing zakończona!")
    print("Zapisano wykresy w bieżącym katalogu.")

if __name__ == "__main__":
    main()