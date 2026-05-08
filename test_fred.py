# test_fred.py
# Script de diagnostic : teste chaque code FRED un par un
from fredapi import Fred
from dotenv import load_dotenv
import os

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

series_a_tester = {
    "gdp_france":       "CLVMNACSCAB1GQFR",
    "gdp_germany":      "CLVMNACSCAB1GQDE",
    "gdp_italy":        "CLVMNACSCAB1GQIT",
    "gdp_spain":        "CLVMNACSCAB1GQES",
    "gdp_eurozone":     "CLVMNACSCAB1GQEA19",
    "unemployment_fr":  "LRHUTTTTFRM156S",
    "unemployment_de":  "LRHUTTTTDEM156S",
    "unemployment_it":  "LRHUTTTTITM156S",
    "unemployment_es":  "LRHUTTTTESM156S",
    "unemployment_ea":  "LRHUTTTTEZM156S",
    "inflation_fr":     "FRACPIALLMINMEI",
    "inflation_de":     "DEUCPIALLMINMEI",
    "inflation_it":     "ITACPIALLMINMEI",
    "inflation_es":     "ESPCIALLMINMEI",
    "ecb_rate":         "ECBMLFR",
    "oat_10y":          "IRLTLT01FRM156N",
    "bund_10y":         "IRLTLT01DEM156N",
    "eurusd":           "DEXUSEU",
    "trade_balance_fr": "XTEXVA01FRM667S",
}

print("\n" + "="*55)
print("DIAGNOSTIC DES SÉRIES FRED")
print("="*55)

ok, erreurs = [], []

for nom, code in series_a_tester.items():
    try:
        s = fred.get_series(code, observation_start="1990-01-01")
        print(f"  ✅ {nom:<22} | {len(s):>4} obs | {str(s.index[0])[:10]} → {str(s.index[-1])[:10]}")
        ok.append(nom)
    except Exception as e:
        print(f"  ❌ {nom:<22} | ERREUR : {e}")
        erreurs.append((nom, code))

print("="*55)
print(f"✅ {len(ok)} séries OK  |  ❌ {len(erreurs)} séries en erreur")
if erreurs:
    print("\nSéries à corriger :")
    for nom, code in erreurs:
        print(f"   → {nom} ({code})")
print("="*55 + "\n")