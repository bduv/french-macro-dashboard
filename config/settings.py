# ============================================================
# config/settings.py
# Paramètres globaux du projet French Macro Dashboard
# ============================================================

import os
from dotenv import load_dotenv

# Charge les variables depuis le fichier .env
load_dotenv()

# ── Clés API ────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY")

# ── Période d'analyse ────────────────────────────────────────
START_DATE = os.getenv("START_DATE", "1990-01-01")
END_DATE   = os.getenv("END_DATE",   "2024-12-31")

# ── Pays analysés ────────────────────────────────────────────
COUNTRIES = {
    "France":     "FRA",
    "Allemagne":  "DEU",
    "Italie":     "ITA",
    "Espagne":    "ESP",
    "Zone Euro":  "EMU",
}

# ── Palette de couleurs du dashboard ─────────────────────────
# Inspirée des couleurs officielles de la Banque de France
COLORS = {
    "France":    "#0831BA",   # Bleu France
    "Allemagne": "#FFCE00",   # Jaune Allemagne
    "Italie":    "#009246",   # Vert Italie
    "Espagne":   "#c60b1e",   # Rouge Espagne
    "Zone Euro": "#021336",   # Bleu UE
    "positive":  "#2ecc71",   # Vert pour valeurs positives
    "negative":  "#e74c3c",   # Rouge pour valeurs négatives
    "neutral":   "#95a5a6",   # Gris neutre
}

# ── Mapping des séries FRED ────────────────────────────────────────
# FRED = base de données de la Fed. Chaque série a un code unique.
FRED_SERIES = {
    # ── PIB réel CVS-CJO (indice de volume, base OCDE) ──────────
    # Source : OCDE via FRED — données trimestrielles CVS
    "gdp_france":       "CLVMNACSCAB1GQFR",
    "gdp_germany":      "CLVMNACSCAB1GQDE",
    "gdp_italy":        "CLVMNACSCAB1GQIT",
    "gdp_spain":        "CLVMNACSCAB1GQES",
    "gdp_eurozone":     "CLVMNACSCAB1GQEA19",

    # ── Chômage harmonisé OCDE (%) ────────────────────────────────
    "unemployment_fr":  "LRHUTTTTFRM156S",
    "unemployment_de":  "LRHUTTTTDEM156S",
    "unemployment_it":  "LRHUTTTTITM156S",
    "unemployment_es":  "LRHUTTTTESM156S",
    "unemployment_ea":  "LRHUTTTTEZM156S",

    # ── INFLATION : IPCH Eurostat (glissement annuel %) ───────────
    # IPCH = Indice des Prix à la Consommation Harmonisé
    # OBLIGATOIRE pour comparaisons inter-pays zone euro
    # Source : Eurostat via FRED
    "inflation_fr":     "CP0000FRM086NEST",   # IPCH France
    "inflation_de":     "CP0000DEM086NEST",   # IPCH Allemagne
    "inflation_it":     "CP0000ITM086NEST",   # IPCH Italie
    "inflation_es":     "CP0000ESM086NEST",   # IPCH Espagne
    "inflation_ea":     "CP0000EZ19M086NEST", # IPCH Zone Euro

    # ── Taux directeur BCE (taux de dépôt) ────────────────────────
    # ECBDFR = taux de facilité de dépôt BCE (le taux de référence
    # depuis 2022 pour la transmission de la politique monétaire)
    "ecb_rate":         "ECBDFR",

    # ── Taux souverains 10 ans ────────────────────────────────────
    "oat_10y":          "IRLTLT01FRM156N",   # OAT France
    "bund_10y":         "IRLTLT01DEM156N",   # Bund Allemagne

    # ── Taux de change ────────────────────────────────────────────
    "eurusd":           "DEXUSEU",

    # ── Balance commerciale France ────────────────────────────────
    "trade_balance_fr": "XTEXVA01FRM667S",
}

# ── Mapping World Bank (indicateurs annuels) ──────────────────
# wbgapi utilise ces codes pour récupérer les données
WORLDBANK_INDICATORS = {
    # Dette publique brute (% PIB)
    "public_debt_gdp":   "GC.DOD.TOTL.GD.ZS",

    # Solde budgétaire (% PIB) — déficit si négatif
    "fiscal_balance":    "GC.BAL.CASH.GD.ZS",

    # Balance courante (% PIB)
    "current_account":   "BN.CAB.XOKA.GD.ZS",

    # Dette des ménages (% revenu disponible brut)
    "household_debt":    "FS.AST.HHLD.GD.ZS",

    # Coûts unitaires du travail (indice)
    "unit_labor_cost":   "NE.EXP.GNFS.ZS",  # proxy export competitiveness

    # PIB par habitant en PPA (USD)
    "gdp_per_capita":    "NY.GDP.PCAP.PP.CD",
}

# ── Paramètres des modèles économétriques ─────────────────────
MODEL_CONFIG = {
    "var_lags":          4,    # Nombre de retards pour le modèle VAR
    "forecast_horizon":  8,    # Horizon de prévision (trimestres)
    "confidence_level":  0.90, # Intervalle de confiance (90%)
    "adf_significance":  0.05, # Seuil de significativité test ADF
}