# ============================================================
# src/data_loader.py
# Téléchargement de toutes les données macroéconomiques
# Sources : FRED (Fed), World Bank, OCDE (via FRED)
# ============================================================

import pandas as pd
import numpy as np
import wbgapi as wb
import streamlit as st
from fredapi import Fred
from datetime import datetime

# On importe notre fichier de configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FRED_API_KEY, START_DATE, END_DATE,
    FRED_SERIES, WORLDBANK_INDICATORS, COUNTRIES
)


# ── Connexion à l'API FRED ─────────────────────────────────────
def get_fred_client():
    """
    Initialise la connexion à l'API FRED.
    Retourne un objet Fred prêt à être utilisé.
    """
    if not FRED_API_KEY:
        raise ValueError(
            "❌ Clé FRED manquante ! "
            "Ajoute FRED_API_KEY dans ton fichier .env"
        )
    return Fred(api_key=FRED_API_KEY)


# ══════════════════════════════════════════════════════════════
# SECTION 1 : Données FRED (PIB, chômage, inflation, taux)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache 1 heure pour éviter les appels répétés
def load_fred_series(series_id: str, series_name: str) -> pd.Series:
    """
    Télécharge une seule série depuis FRED.
    
    Arguments :
        series_id   : Code FRED de la série (ex: "CLVMNACSCAB1GQFR")
        series_name : Nom lisible (ex: "PIB France")
    
    Retourne :
        Une pd.Series avec l'index en dates et le nom donné
    """
    try:
        fred = get_fred_client()
        serie = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE
        )
        serie.name = series_name
        print(f"  ✅ Chargé : {series_name} ({len(serie)} observations)")
        return serie

    except Exception as e:
        print(f"  ⚠️  Erreur pour {series_name} ({series_id}) : {e}")
        # Retourne une série vide plutôt que de planter tout le dashboard
        return pd.Series(name=series_name, dtype=float)


@st.cache_data(ttl=3600)
def load_all_fred_data() -> dict:
    """
    Télécharge TOUTES les séries FRED définies dans settings.py.
    
    Retourne :
        Un dictionnaire {nom_serie: pd.Series}
    """
    print("\n🔄 Chargement des données FRED...")
    data = {}

    for nom, code in FRED_SERIES.items():
        data[nom] = load_fred_series(code, nom)

    print(f"✅ {len(data)} séries FRED chargées.\n")
    return data


# ══════════════════════════════════════════════════════════════
# SECTION 2 : Données World Bank (dette, balance courante...)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400)  # Cache 24h (données annuelles, changent peu)
def load_worldbank_data() -> dict:
    """
    Télécharge les indicateurs annuels depuis la Banque Mondiale.
    
    Retourne :
        Un dictionnaire {nom_indicateur: pd.DataFrame}
        Chaque DataFrame a les pays en colonnes, les années en index
    """
    print("\n🔄 Chargement des données World Bank...")
    
    # Codes des pays pour la World Bank
    pays_wb = list(COUNTRIES.values())  # ["FRA", "DEU", "ITA", "ESP", "EMU"]
    
    # Conversion des années (World Bank utilise des entiers)
    annee_debut = int(START_DATE[:4])
    annee_fin   = int(END_DATE[:4])
    
    data_wb = {}

    for nom_indicateur, code_wb in WORLDBANK_INDICATORS.items():
        try:
            # Téléchargement via wbgapi
            df = wb.data.DataFrame(
                code_wb,
                economy=pays_wb,
                time=range(annee_debut, annee_fin + 1),
                skipBlanks=True,
                labels=False
            )
            
            # Nettoyage : on transpose pour avoir années en lignes, pays en colonnes
            df = df.T
            df.index = pd.to_datetime(df.index.astype(str), format='YR%Y')
            df.columns = [
                next((k for k, v in COUNTRIES.items() if v == c), c)
                for c in df.columns
            ]
            df = df.sort_index()
            
            data_wb[nom_indicateur] = df
            print(f"  ✅ Chargé : {nom_indicateur}")

        except Exception as e:
            print(f"  ⚠️  Erreur World Bank pour {nom_indicateur} : {e}")
            data_wb[nom_indicateur] = pd.DataFrame()

    print(f"✅ {len(data_wb)} indicateurs World Bank chargés.\n")
    return data_wb


# ══════════════════════════════════════════════════════════════
# SECTION 3 : Construction des DataFrames principaux
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def build_gdp_dataframe() -> pd.DataFrame:
    """
    Construit le DataFrame du PIB pour les 5 zones géographiques.
    Calcule aussi le taux de croissance trimestriel et annuel.
    
    Retourne :
        df_gdp avec colonnes : France, Allemagne, Italie, Espagne, Zone Euro
        + colonnes de croissance
    """
    fred_data = load_all_fred_data()
    
    # Assemblage des séries PIB en un seul DataFrame
    df_gdp = pd.DataFrame({
        "France":    fred_data.get("gdp_france",    pd.Series(dtype=float)),
        "Allemagne": fred_data.get("gdp_germany",   pd.Series(dtype=float)),
        "Italie":    fred_data.get("gdp_italy",     pd.Series(dtype=float)),
        "Espagne":   fred_data.get("gdp_spain",     pd.Series(dtype=float)),
        "Zone Euro": fred_data.get("gdp_eurozone",  pd.Series(dtype=float)),
    })
    
    # Calcul croissance trimestrielle (variation % par rapport au trimestre précédent)
    for pays in df_gdp.columns:
        df_gdp[f"{pays}_growth_q"] = df_gdp[pays].pct_change() * 100
    
    # Calcul croissance annuelle (variation % par rapport à même trimestre année N-1)
    for pays in ["France", "Allemagne", "Italie", "Espagne", "Zone Euro"]:
        df_gdp[f"{pays}_growth_y"] = df_gdp[pays].pct_change(4) * 100
    
    return df_gdp.dropna(how="all")


@st.cache_data(ttl=3600)
def build_unemployment_dataframe() -> pd.DataFrame:
    """
    Construit le DataFrame du chômage pour les 5 zones.
    """
    fred_data = load_all_fred_data()
    
    df_unemp = pd.DataFrame({
        "France":    fred_data.get("unemployment_fr", pd.Series(dtype=float)),
        "Allemagne": fred_data.get("unemployment_de", pd.Series(dtype=float)),
        "Italie":    fred_data.get("unemployment_it", pd.Series(dtype=float)),
        "Espagne":   fred_data.get("unemployment_es", pd.Series(dtype=float)),
        "Zone Euro": fred_data.get("unemployment_ea", pd.Series(dtype=float)),
    })
    
    return df_unemp.dropna(how="all")


@st.cache_data(ttl=3600)
def build_inflation_dataframe() -> pd.DataFrame:
    """
    IPCH Eurostat : les séries CP0000...NEST sont des INDICES (base 2015=100).
    On calcule le glissement annuel (pct_change(12) × 100) avant tout affichage.
    """
    fred_data = load_all_fred_data()

    pays_series = {
        "France":    fred_data.get("inflation_fr", pd.Series(dtype=float)),
        "Allemagne": fred_data.get("inflation_de", pd.Series(dtype=float)),
        "Italie":    fred_data.get("inflation_it", pd.Series(dtype=float)),
        "Espagne":   fred_data.get("inflation_es", pd.Series(dtype=float)),
        "Zone Euro": fred_data.get("inflation_ea", pd.Series(dtype=float)),
    }

    df_inf = pd.DataFrame()
    for pays, serie in pays_series.items():
        if not serie.empty:
            # Glissement annuel sur l'indice brut
            df_inf[pays] = serie.pct_change(periods=12) * 100
        else:
            df_inf[pays] = pd.Series(dtype=float)

    return df_inf.dropna(how="all")


@st.cache_data(ttl=3600)
def build_rates_dataframe() -> pd.DataFrame:
    """
    Construit le DataFrame des taux d'intérêt.
    Calcule aussi le spread OAT-Bund (indicateur de risque souverain).
    """
    fred_data = load_all_fred_data()
    
    df_rates = pd.DataFrame({
        "BCE (taux directeur)":   fred_data.get("ecb_rate",  pd.Series(dtype=float)),
        "OAT 10 ans (France)":    fred_data.get("oat_10y",   pd.Series(dtype=float)),
        "Bund 10 ans (Allemagne)":fred_data.get("bund_10y",  pd.Series(dtype=float)),
        "EUR/USD":                fred_data.get("eurusd",    pd.Series(dtype=float)),
    })
    
    # Spread OAT-Bund = prime de risque de la France vs Allemagne
    # Un spread élevé signifie que les marchés perçoivent la France comme plus risquée
    if "OAT 10 ans (France)" in df_rates and "Bund 10 ans (Allemagne)" in df_rates:
        df_rates["Spread OAT-Bund (pb)"] = (
            df_rates["OAT 10 ans (France)"] - df_rates["Bund 10 ans (Allemagne)"]
        ) * 100  # En points de base (100 pb = 1%)
    
    return df_rates.dropna(how="all")


@st.cache_data(ttl=86400)
def build_fiscal_dataframe() -> pd.DataFrame:
    """
    Construit le DataFrame des finances publiques (World Bank, annuel).
    Couvre : dette publique, déficit, balance courante.
    """
    wb_data = load_worldbank_data()
    
    return {
        "dette_publique":  wb_data.get("public_debt_gdp",  pd.DataFrame()),
        "solde_budgetaire":wb_data.get("fiscal_balance",   pd.DataFrame()),
        "balance_courante":wb_data.get("current_account",  pd.DataFrame()),
        "dette_menages":   wb_data.get("household_debt",   pd.DataFrame()),
    }


# ══════════════════════════════════════════════════════════════
# SECTION 4 : Chargement global (tout en une fois)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_all_data() -> dict:
    """
    Fonction principale : charge TOUTES les données en une seule fois.
    C'est cette fonction qu'on appellera depuis les pages du dashboard.
    
    Retourne :
        Un dictionnaire avec toutes les données organisées par thème
    """
    print("\n" + "="*60)
    print("🚀 CHARGEMENT COMPLET DES DONNÉES MACRO")
    print("="*60)
    
    data = {
        "gdp":          build_gdp_dataframe(),
        "unemployment": build_unemployment_dataframe(),
        "inflation":    build_inflation_dataframe(),
        "rates":        build_rates_dataframe(),
        "fiscal":       build_fiscal_dataframe(),
        "last_update":  datetime.now().strftime("%d/%m/%Y à %H:%M"),
    }
    
    print("\n✅ Toutes les données chargées avec succès !")
    print(f"   PIB           : {len(data['gdp'])} observations")
    print(f"   Chômage       : {len(data['unemployment'])} observations")
    print(f"   Inflation     : {len(data['inflation'])} observations")
    print(f"   Taux          : {len(data['rates'])} observations")
    print("="*60 + "\n")
    
    return data
@st.cache_data(ttl=3600)
def build_gdp_dataframe() -> pd.DataFrame:
    """
    PIB réel — deux mesures de croissance :
    1. Glissement annuel trimestriel (pct_change(4)) — pour graphiques
    2. Croissance annuelle moyenne — pour comparaison avec chiffres INSEE
    """
    from src.data_processor import (
        compute_gdp_growth_quarterly_yoy,
        compute_gdp_growth_annual_average
    )
    fred_data = load_all_fred_data()

    pays = {
        "France":    "gdp_france",
        "Allemagne": "gdp_germany",
        "Italie":    "gdp_italy",
        "Espagne":   "gdp_spain",
        "Zone Euro": "gdp_eurozone",
    }

    df_gdp = pd.DataFrame()

    for label, key in pays.items():
        serie = fred_data.get(key, pd.Series(dtype=float))
        if not serie.empty:
            df_gdp[label]                    = serie
            df_gdp[f"{label}_growth_q"]      = serie.pct_change() * 100
            df_gdp[f"{label}_growth_y"]      = compute_gdp_growth_quarterly_yoy(serie)
            df_gdp[f"{label}_growth_annual"] = compute_gdp_growth_annual_average(serie).reindex(serie.index, method="ffill")

    return df_gdp.dropna(how="all")