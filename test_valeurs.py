# test_valeurs.py
# Vérifie que les valeurs affichées correspondent aux sources officielles

import sys
sys.path.append(".")
from src.data_loader import build_inflation_dataframe, build_gdp_dataframe
from src.data_loader import build_unemployment_dataframe, build_rates_dataframe

print("\n" + "="*55)
print("VALIDATION DES VALEURS — SOURCES OFFICIELLES")
print("="*55)

# PIB
gdp = build_gdp_dataframe()
croissance = gdp["France_growth_y"].dropna()
print(f"\n📊 PIB France — dernière croissance annuelle :")
print(f"   Valeur dashboard : {croissance.iloc[-1]:.1f}%")
print(f"   Valeur INSEE 2024 : +1.2%")

# Chômage
unemp = build_unemployment_dataframe()
print(f"\n👥 Chômage France — dernière valeur :")
print(f"   Valeur dashboard : {unemp['France'].dropna().iloc[-1]:.1f}%")
print(f"   Valeur INSEE T4 2024 : 7.3%")

# Inflation
inf = build_inflation_dataframe()
print(f"\n💰 Inflation IPCH France — dernière valeur :")
print(f"   Valeur dashboard : {inf['France'].dropna().iloc[-1]:.1f}%")
print(f"   Valeur INSEE déc. 2024 (IPCH) : +1.8%")

# Taux
rates = build_rates_dataframe()
print(f"\n📉 OAT 10 ans France — dernière valeur :")
print(f"   Valeur dashboard : {rates['OAT 10 ans (France)'].dropna().iloc[-1]:.2f}%")
print(f"   Valeur référence déc. 2024 : ~3.20%")

if "Spread OAT-Bund (pb)" in rates.columns:
    print(f"\n📉 Spread OAT-Bund — dernière valeur :")
    print(f"   Valeur dashboard : {rates['Spread OAT-Bund (pb)'].dropna().iloc[-1]:.0f} pb")
    print(f"   Valeur référence fin 2024 : ~70-80 pb")

print("="*55 + "\n")