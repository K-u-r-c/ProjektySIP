import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

# Pobieranie danych
housing = fetch_california_housing()

# Konwersja na DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MEDV'] = housing.target  # Dodanie zmiennej celu (mediany wartości domów)

# Wyświetlenie szczegółów o danych
print("INFORMACJE O ZBIORZE DANYCH CALIFORNIA HOUSING")
print("=============================================")
print(f"Liczba obserwacji: {df.shape[0]}")
print(f"Liczba zmiennych: {df.shape[1]}")
print("\nOPIS ZMIENNYCH:")
print("-------------")
print("MedInc      - mediana dochodów gospodarstw domowych w bloku (w 10,000 USD)")
print("HouseAge    - mediana wieku domów w bloku")
print("AveRooms    - średnia liczba pokoi na gospodarstwo domowe")
print("AveBedrms   - średnia liczba sypialni na gospodarstwo domowe")
print("Population  - populacja w bloku")
print("AveOccup    - średnia liczba mieszkańców na gospodarstwo domowe")
print("Latitude    - szerokość geograficzna bloku")
print("Longitude   - długość geograficzna bloku")
print("MEDV        - mediana wartości domów w bloku (w 100,000 USD)")

# Wyświetlenie pierwszych 10 wierszy danych
print("\nPIERWSZYCH 10 WIERSZY DANYCH:")
print("-------------------------")
print(df.head(10).round(3))

# Wyświetlenie podstawowych statystyk
print("\nPODSTAWOWE STATYSTYKI:")
print("-------------------")
print(df.describe().round(3))

# Zapisanie danych do pliku CSV
csv_file = "california_housing_data.csv"
df.to_csv(csv_file, index=False)
print(f"\nZapisano dane do pliku {csv_file}")

# Stworzenie atrakcyjnej tabeli z wartościami statystycznymi
def create_pretty_table():
    # Obliczenie dodatkowych statystyk dla każdej kolumny
    stats = pd.DataFrame({
        'Min': df.min(),
        'Max': df.max(),
        'Średnia': df.mean(),
        'Mediana': df.median(),
        'Odch. std.': df.std(),
        'Skośność': df.skew(),
        '% brakujących': df.isna().mean() * 100
    }).round(3)
    
    # Wyświetlenie statystyk w formie tabeli
    print("\nSTATYSTYKI SZCZEGÓŁOWE:")
    print("---------------------")
    print(stats)
    
    # Stworzenie wykresu z przeglądem danych
    plt.figure(figsize=(14, 10))
    
    # Wykres rozrzutu dla lokalizacji (długość i szerokość geograficzna)
    # z kolorem oznaczającym wartość domów
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                c=df['MEDV'], cmap='viridis', 
                alpha=0.6, s=15, edgecolor='none')
    plt.colorbar(scatter, label='Wartość domów (MEDV)')
    plt.title('Lokalizacja nieruchomości i ich wartość')
    plt.xlabel('Długość geograficzna')
    plt.ylabel('Szerokość geograficzna')
    plt.grid(True, alpha=0.3)
    
    # Wykres pudełkowy dla głównych zmiennych
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']])
    plt.title('Rozkład głównych zmiennych')
    plt.grid(True, alpha=0.3)
    
    # Histogram wartości domów
    plt.subplot(2, 2, 3)
    plt.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Rozkład wartości domów (MEDV)')
    plt.xlabel('Wartość (w 100,000 USD)')
    plt.ylabel('Liczba obserwacji')
    plt.grid(True, alpha=0.3)
    
    # Heatmap korelacji
    plt.subplot(2, 2, 4)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Macierz korelacji')
    
    plt.tight_layout()
    plt.savefig('california_housing_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

# Wywołanie funkcji tworzącej atrakcyjną tabelę statystyczną
create_pretty_table()

# Funkcja do losowego próbkowania i wyświetlania przykładowych rekordów
def sample_records(n=20):
    """Wyświetla losowo wybrane rekordy z danymi"""
    sampled = df.sample(n=n, random_state=42)
    print(f"\nLOSOWA PRÓBKA {n} REKORDÓW:")
    print("-" * (14 + len(str(n))))
    print(sampled.round(3))
    return sampled

# Wyświetlenie losowej próbki rekordów
sample_records(20)

# Dodatkowa analiza interpretacyjna
print("\nINTERPRETACJA GŁÓWNYCH CECH:")
print("--------------------------")

# Analiza mediany dochodów
income_groups = pd.cut(df['MedInc'], bins=5)
income_analysis = df.groupby(income_groups)['MEDV'].agg(['mean', 'count']).reset_index()
income_analysis.columns = ['Zakres dochodów', 'Średnia wartość domu', 'Liczba obserwacji']
print("\nWartość domów według grup dochodowych:")
print(income_analysis.round(3))

# Analiza wieku domów
age_groups = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 40, 100])
age_analysis = df.groupby(age_groups)['MEDV'].agg(['mean', 'count']).reset_index()
age_analysis.columns = ['Zakres wieku domów', 'Średnia wartość domu', 'Liczba obserwacji']
print("\nWartość domów według wieku budynków:")
print(age_analysis.round(3))

# Analiza dla zagęszczenia mieszkańców
occup_groups = pd.cut(df['AveOccup'], bins=[0, 2, 3, 4, 5, df['AveOccup'].max()])
occup_analysis = df.groupby(occup_groups)['MEDV'].agg(['mean', 'count']).reset_index()
occup_analysis.columns = ['Średnia liczba mieszkańców', 'Średnia wartość domu', 'Liczba obserwacji']
print("\nWartość domów według zagęszczenia mieszkańców:")
print(occup_analysis.round(3))

print("\nUWAGA: Wartości MEDV są podane w 100,000 USD")
print("Pełny zbiór danych zapisano do pliku CSV:", csv_file)
print("Wizualizację przeglądu danych zapisano do pliku: california_housing_overview.png")