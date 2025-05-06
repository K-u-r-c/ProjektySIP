import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, normaltest, jarque_bera
from scipy.stats import ttest_1samp, chi2
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Load and describe the data
def load_and_describe_data():
    """Load California Housing dataset and provide description"""
    print("1. OPIS DANYCH I ICH ŹRÓDŁO")
    print("=" * 50)
    
    # Fetch the dataset
    housing = fetch_california_housing()
    
    # Convert to DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MEDV'] = housing.target  # Adding target variable (median house value)
    
    # Dataset description
    print("Zbiór danych: California Housing Dataset")
    print("Źródło: Biblioteka scikit-learn (sklearn.datasets)")
    print("\nOpis: Ten zbiór danych zawiera informacje o cenach domów w różnych")
    print("      regionach Kalifornii, wraz z czynnikami demograficznymi i lokalizacyjnymi.")
    print(f"\nLiczba obserwacji: {df.shape[0]}")
    print(f"Liczba zmiennych: {df.shape[1]}")
    
    print("\nOPIS ZMIENNYCH:")
    print("- MedInc:     Mediana dochodów gospodarstw domowych w bloku (w 10,000 USD)")
    print("- HouseAge:   Mediana wieku domów w bloku")
    print("- AveRooms:   Średnia liczba pokoi na gospodarstwo domowe")
    print("- AveBedrms:  Średnia liczba sypialni na gospodarstwo domowe")
    print("- Population: Populacja w bloku")
    print("- AveOccup:   Średnia liczba mieszkańców na gospodarstwo domowe")
    print("- Latitude:   Szerokość geograficzna bloku")
    print("- Longitude:  Długość geograficzna bloku")
    print("- MEDV:       Mediana wartości domów w bloku (w 100,000 USD)")
    
    # Display first rows
    print("\nPierwszych 5 wierszy danych:")
    print(df.head().round(3))
    
    return df

# 2. Point estimation of distribution parameters
def point_estimation(df):
    """Calculate point estimates of distribution parameters"""
    print("\n\n2. ESTYMACJA PUNKTOWA PARAMETRÓW ROZKŁADU")
    print("=" * 50)
    
    # Basic statistics
    basic_stats = df.describe().round(3)
    print("Podstawowe statystyki:")
    print(basic_stats)
    
    # Additional statistics
    stats_df = pd.DataFrame(index=df.columns)
    
    # Mean
    stats_df['Średnia'] = df.mean().round(3)
    
    # Median
    stats_df['Mediana'] = df.median().round(3)
    
    # Mode (most common value) - handle with try-except in case mode() returns empty DataFrame
    try:
        stats_df['Moda'] = df.mode().iloc[0].round(3)
    except:
        # If mode fails, use median as fallback
        stats_df['Moda'] = df.median().round(3)
    
    # Standard deviation
    stats_df['Odch. std.'] = df.std().round(3)
    
    # Mean absolute deviation (manually calculated since df.mad() is deprecated)
    stats_df['Odch. przeciętne'] = (df - df.mean()).abs().mean().round(3)
    
    # Variance
    stats_df['Wariancja'] = df.var().round(3)
    
    # Skewness
    stats_df['Skośność'] = df.skew().round(3)
    
    # Kurtosis
    stats_df['Kurtoza'] = df.kurtosis().round(3)
    
    # Interquartile range (IQR)
    stats_df['IQR'] = (df.quantile(0.75) - df.quantile(0.25)).round(3)
    
    # 10th percentile
    stats_df['Q10'] = df.quantile(0.1).round(3)
    
    # 25th percentile
    stats_df['Q25'] = df.quantile(0.25).round(3)
    
    # 75th percentile
    stats_df['Q75'] = df.quantile(0.75).round(3)
    
    # 90th percentile
    stats_df['Q90'] = df.quantile(0.9).round(3)
    
    # Coefficient of variation
    stats_df['Wsp. zmienności'] = (df.std() / df.mean()).round(3)
    
    print("\nDodatkowe statystyki:")
    print(stats_df.T)
    
    return stats_df

# 3. Interval estimation and bootstrap
def interval_estimation(df):
    """Calculate confidence intervals and bootstrap estimates"""
    print("\n\n3. ESTYMACJA PRZEDZIAŁOWA I BOOTSTRAP")
    print("=" * 50)
    
    # Focus on the target variable MEDV for demonstration
    target = 'MEDV'
    data = df[target]
    n = len(data)
    
    # Sample mean and standard deviation
    mean = data.mean()
    std = data.std()
    
    # 3.1 Confidence interval for mean (parametric)
    # Using t-distribution because population variance is unknown
    alpha = 0.05  # 95% confidence level
    t_crit = stats.t.ppf(1 - alpha/2, n-1)
    
    mean_ci_lower = mean - t_crit * std / np.sqrt(n)
    mean_ci_upper = mean + t_crit * std / np.sqrt(n)
    
    print(f"Przedział ufności dla średniej ({target}, 95%):")
    print(f"  Metoda parametryczna: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]")
    
    # 3.2 Confidence interval for variance (parametric)
    # Using chi-square distribution
    chi2_lower = stats.chi2.ppf(alpha/2, n-1)
    chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)
    
    var = data.var()
    var_ci_lower = (n-1) * var / chi2_upper
    var_ci_upper = (n-1) * var / chi2_lower
    
    print(f"Przedział ufności dla wariancji ({target}, 95%):")
    print(f"  Metoda parametryczna: [{var_ci_lower:.4f}, {var_ci_upper:.4f}]")
    
    # 3.3 Bootstrap confidence interval for mean (non-parametric)
    np.random.seed(42)
    n_bootstrap = 5000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentile-based confidence interval
    bootstrap_ci_lower = np.percentile(bootstrap_means, 2.5)
    bootstrap_ci_upper = np.percentile(bootstrap_means, 97.5)
    
    print(f"Przedział ufności dla średniej ({target}, 95%):")
    print(f"  Metoda bootstrap: [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")
    
    # Visualize bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Średnia próby ({mean:.4f})')
    plt.axvline(bootstrap_ci_lower, color='green', linestyle=':', linewidth=2, 
                label=f'Dolna granica CI ({bootstrap_ci_lower:.4f})')
    plt.axvline(bootstrap_ci_upper, color='green', linestyle=':', linewidth=2,
                label=f'Górna granica CI ({bootstrap_ci_upper:.4f})')
    plt.title(f'Rozkład bootstrapowy średniej dla {target}')
    plt.xlabel('Wartość średniej')
    plt.ylabel('Częstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bootstrap_distribution.png', dpi=300)
    
    return {
        'mean_ci': (mean_ci_lower, mean_ci_upper),
        'var_ci': (var_ci_lower, var_ci_upper),
        'bootstrap_ci': (bootstrap_ci_lower, bootstrap_ci_upper)
    }

# 4. Various plots
def create_plots(df):
    """Create various plots for data visualization"""
    print("\n\n4. WIZUALIZACJA DANYCH")
    print("=" * 50)
    
    # 4.1 Histograms for all variables
    print("Tworzenie histogramów dla wszystkich zmiennych...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Histogram: {col}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=300)
    
    # 4.2 Q-Q plots for all variables
    print("Tworzenie wykresów kwantyl-kwantyl...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        qqplot(df[col], line='s', ax=axes[i])
        axes[i].set_title(f'Wykres Q-Q: {col}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qqplots.png', dpi=300)
    
    # 4.3 Box plots for all variables
    print("Tworzenie boxplotów...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=300)
    
    # 4.4 Pair plot for a subset of variables
    print("Tworzenie wykresu par (pairplot)...")
    subset_df = df[['MedInc', 'HouseAge', 'AveRooms', 'MEDV']]
    g = sns.pairplot(subset_df, diag_kind='kde', corner=True)
    g.fig.suptitle('Pairplot dla wybranych zmiennych', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=300)
    
    # 4.5 Correlation heatmap
    print("Tworzenie macierzy korelacji...")
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Macierz korelacji')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    
    # 4.6 Scatter plot with regression line for MedInc vs MEDV
    print("Tworzenie wykresu rozrzutu z linią regresji...")
    plt.figure(figsize=(10, 6))
    sns.regplot(x='MedInc', y='MEDV', data=df, scatter_kws={'alpha':0.5})
    plt.title('Zależność wartości domów od mediany dochodów')
    plt.xlabel('Mediana dochodów (MedInc)')
    plt.ylabel('Mediana wartości domów (MEDV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regplot.png', dpi=300)
    
    # 4.7 Violin plots
    print("Tworzenie wykresów skrzypcowych...")
    # Create categorical bins for MedInc to use in violin plot
    df['MedInc_cat'] = pd.qcut(df['MedInc'], 5, labels=['Bardzo niski', 'Niski', 'Średni', 'Wysoki', 'Bardzo wysoki'])
    
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='MedInc_cat', y='MEDV', data=df)
    plt.title('Rozkład wartości domów według kategorii dochodów')
    plt.xlabel('Kategoria dochodów')
    plt.ylabel('Mediana wartości domów (MEDV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('violinplot.png', dpi=300)
    
    print("Wszystkie wykresy zostały zapisane jako pliki PNG.")

# 5. Test for normality
def test_normality(df):
    """Test if variables come from normal distribution"""
    print("\n\n5. TEST NORMALNOŚCI ROZKŁADU")
    print("=" * 50)
    
    # Make a copy of the dataframe to avoid modifying the original
    test_df = df.copy()
    
    # Drop any categorical columns that might have been added
    for col in test_df.columns:
        if test_df[col].dtype == 'object' or test_df[col].dtype.name == 'category':
            test_df = test_df.drop(columns=[col])
    
    normality_results = pd.DataFrame(
        index=test_df.columns,
        columns=['Shapiro-Wilk p-value', 'D\'Agostino p-value', 'Jarque-Bera p-value', 'Is Normal?']
    )
    
    alpha = 0.05  # significance level
    
    for col in test_df.columns:
        # Calculate p-values for different normality tests
        # For large datasets, we might get very small p-values even for minor deviations
        try:
            shapiro_test = shapiro(test_df[col])
            dagostino_test = normaltest(test_df[col])
            jarque_bera_test = jarque_bera(test_df[col])
            
            # Store p-values
            normality_results.loc[col, 'Shapiro-Wilk p-value'] = shapiro_test[1]
            normality_results.loc[col, 'D\'Agostino p-value'] = dagostino_test[1]
            normality_results.loc[col, 'Jarque-Bera p-value'] = jarque_bera_test[1]
            
            # Determine if normal based on all tests
            is_normal = (shapiro_test[1] > alpha) or (dagostino_test[1] > alpha) or (jarque_bera_test[1] > alpha)
            normality_results.loc[col, 'Is Normal?'] = "Tak" if is_normal else "Nie"
        except Exception as e:
            print(f"Uwaga: Wystąpił błąd przy testowaniu normalności dla zmiennej {col}: {e}")
            normality_results.loc[col, 'Shapiro-Wilk p-value'] = np.nan
            normality_results.loc[col, 'D\'Agostino p-value'] = np.nan
            normality_results.loc[col, 'Jarque-Bera p-value'] = np.nan
            normality_results.loc[col, 'Is Normal?'] = "Błąd testu"
    
    print("Wyniki testów normalności:")
    print(normality_results.round(6))
    
    # Create comparative normal probability plots
    print("\nTworzenie wykresów porównawczych z rozkładem normalnym...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Use only numeric columns for plotting
    numeric_cols = test_df.columns
    
    for i, col in enumerate(numeric_cols):
        if i >= len(axes):  # Ensure we don't exceed the number of subplot axes
            break
            
        # Get the data
        data = test_df[col]
        
        # Fit a normal distribution
        mu, std = stats.norm.fit(data)
        
        # Create a histogram with normal distribution overlay
        sns.histplot(data, kde=False, stat='density', ax=axes[i], color='skyblue', alpha=0.6)
        
        # Generate points for the normal distribution curve
        x = np.linspace(data.min(), data.max(), 100)
        p = stats.norm.pdf(x, mu, std)
        
        # Plot the normal distribution
        axes[i].plot(x, p, 'r-', linewidth=2)
        
        # Add title and labels
        axes[i].set_title(f'{col}: Porównanie z rozkładem normalnym')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Gęstość')
        
        # Add text with test results
        is_normal = normality_results.loc[col, 'Is Normal?'] if col in normality_results.index else "Brak danych"
        text = f"Rozkład normalny: {is_normal}"
        axes[i].text(0.05, 0.95, text, transform=axes[i].transAxes, 
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # If we have fewer variables than subplots, hide the extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('normality_tests.png', dpi=300)
    
    return normality_results

# 6. Statistical tests for mean and variance
def statistical_tests(df):
    """Perform statistical tests for mean and variance"""
    print("\n\n6. TESTY STATYSTYCZNE DLA ŚREDNIEJ I WARIANCJI")
    print("=" * 50)
    
    # Focus on MEDV for demonstration
    target = 'MEDV'
    data = df[target]
    
    # 6.1 One-sample t-test for mean
    # Let's test if the mean house value is equal to 2.0 (hypothetical value)
    hypothetical_mean = 2.0
    t_stat, p_value = ttest_1samp(data, hypothetical_mean)
    
    print(f"Test t dla średniej zmiennej {target}:")
    print(f"  Hipoteza H0: średnia = {hypothetical_mean}")
    print(f"  Hipoteza H1: średnia ≠ {hypothetical_mean}")
    print(f"  Statystyka testowa t: {t_stat:.4f}")
    print(f"  Wartość p: {p_value:.8f}")
    print(f"  Wniosek: {'Odrzucamy' if p_value < 0.05 else 'Nie odrzucamy'} hipotezę H0 na poziomie istotności 0.05")
    
    # 6.2 Chi-square test for variance
    # Let's test if the variance is equal to 1.0 (hypothetical value)
    hypothetical_var = 1.0
    n = len(data)
    observed_var = data.var()
    chi2_stat = (n - 1) * observed_var / hypothetical_var
    
    # Use try-except to handle potential numerical issues
    try:
        p_value_chi2 = 1 - stats.chi2.cdf(chi2_stat, n - 1)  # upper tail test
    except:
        print("  Uwaga: Wystąpił problem z obliczeniem p-value dla testu chi-kwadrat.")
        p_value_chi2 = 0.0  # Default to rejecting null hypothesis
    
    print(f"\nTest chi-kwadrat dla wariancji zmiennej {target}:")
    print(f"  Hipoteza H0: wariancja = {hypothetical_var}")
    print(f"  Hipoteza H1: wariancja > {hypothetical_var}")
    print(f"  Statystyka testowa chi2: {chi2_stat:.4f}")
    print(f"  Wartość p: {p_value_chi2:.8f}")
    print(f"  Wniosek: {'Odrzucamy' if p_value_chi2 < 0.05 else 'Nie odrzucamy'} hipotezę H0 na poziomie istotności 0.05")
    
    # Visualize the data with reference to hypothetical mean
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='skyblue')
    plt.axvline(hypothetical_mean, color='red', linestyle='--', 
                label=f'Hipoteza H0: średnia = {hypothetical_mean}')
    plt.axvline(data.mean(), color='green', linestyle='-', 
                label=f'Obserwowana średnia = {data.mean():.4f}')
    plt.title(f'Rozkład {target} z zaznaczoną średnią hipotezowaną i obserwowaną')
    plt.xlabel(target)
    plt.ylabel('Częstość')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_test_mean.png', dpi=300)
    
    return {
        't_test': {'t_stat': t_stat, 'p_value': p_value},
        'chi2_test': {'chi2_stat': chi2_stat, 'p_value': p_value_chi2}
    }

# 7. Kernel density estimation
def kernel_density_estimation(df):
    """Perform kernel density estimation"""
    print("\n\n7. ESTYMATOR JĄDROWY GĘSTOŚCI")
    print("=" * 50)
    
    # Focus on MEDV for demonstration
    target = 'MEDV'
    data = df[target]
    
    # 7.1 KDE with different bandwidths
    bandwidths = [0.1, 0.3, 0.5, 1.0, 1.5]
    
    plt.figure(figsize=(12, 8))
    
    # Plot histogram as a reference
    sns.histplot(data, kde=False, stat='density', alpha=0.3, color='gray', label='Histogram')
    
    # Plot KDE curves with different bandwidths
    for bw in bandwidths:
        kde = sm.nonparametric.KDEUnivariate(data)
        kde.fit(bw=bw)
        plt.plot(kde.support, kde.density, label=f'KDE (h={bw})')
    
    # Plot normal distribution for comparison
    mu, std = stats.norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label='Rozkład normalny')
    
    plt.title(f'Estymator jądrowy gęstości dla {target}')
    plt.xlabel(target)
    plt.ylabel('Gęstość')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kde.png', dpi=300)
    
    # 7.2 KDE for two variables (bivariate)
    print("\nTworzenie dwuwymiarowego estymatora jądrowego...")
    plt.figure(figsize=(10, 8))
    
    # Choosing MedInc and MEDV for demonstration
    var1 = 'MedInc'
    var2 = 'MEDV'
    
    # Create a 2D KDE plot
    sns.kdeplot(data=df, x=var1, y=var2, fill=True, cmap='viridis', levels=20)
    plt.title(f'Dwuwymiarowy estymator jądrowy gęstości: {var1} vs {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bivariate_kde.png', dpi=300)
    
    print("Wykresy estymatorów jądrowych zostały zapisane.")

# Main function to run the entire analysis
def main():
    """Run the entire statistical analysis"""
    print("\nSTATYSTYCZNA ANALIZA ZBIORU DANYCH CALIFORNIA HOUSING")
    print("=" * 60)
    
    # Load and describe data
    df = load_and_describe_data()
    
    # Point estimation
    stats_df = point_estimation(df)
    
    # Interval estimation and bootstrap
    ci_results = interval_estimation(df)
    
    # Create various plots
    create_plots(df)
    
    # Test for normality
    normality_results = test_normality(df)
    
    # Statistical tests
    test_results = statistical_tests(df)
    
    # Kernel density estimation
    kernel_density_estimation(df)
    
    print("\n\nANALIZA ZAKOŃCZONA SUKCESEM!")
    print("Wszystkie wykresy zostały zapisane jako pliki PNG.")

if __name__ == "__main__":
    main()