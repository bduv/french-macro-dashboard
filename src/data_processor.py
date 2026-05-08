# ============================================================
# src/data_processor.py
# Nettoyage, transformation et enrichissement des données
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st


def resample_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit des données mensuelles en trimestrielles.
    On prend la moyenne du trimestre (standard en macroéconomie).
    
    Pourquoi ? Les données FRED arrivent parfois en mensuel,
    mais nos modèles VAR travaillent en trimestriel.
    """
    return df.resample("QS").mean()


def compute_yoy_change(series: pd.Series) -> pd.Series:
    """
    Calcule la variation en glissement annuel (Year-over-Year).
    Ex : si la valeur en T2-2023 = 100 et T2-2022 = 95,
         le glissement annuel = +5.26%
    """
    return series.pct_change(periods=4) * 100


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcule le Z-score sur fenêtre glissante.
    Utile pour identifier les valeurs aberrantes ou les chocs.
    Z-score > 2 ou < -2 = valeur inhabituelle (> 2 écarts-types)
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std  = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std


def normalize_index(df: pd.DataFrame, base_date: str = None) -> pd.DataFrame:
    """
    Normalise les séries en base 100 à une date donnée.
    Permet de comparer des pays même si leurs PIB sont très différents.
    
    Ex : base_date="2000-01-01" → tout le monde commence à 100 en 2000
    """
    if base_date is None:
        # Par défaut : base 100 à la première observation disponible
        base_values = df.iloc[0]
    else:
        # Trouve la ligne la plus proche de la date voulue
        idx = df.index.get_indexer([pd.Timestamp(base_date)], method="nearest")[0]
        base_values = df.iloc[idx]
    
    return (df / base_values) * 100


def detect_recessions(gdp_growth: pd.Series) -> pd.Series:
    """
    Détecte les périodes de récession selon la règle standard :
    2 trimestres consécutifs de croissance négative.
    
    Retourne une série booléenne : True = récession
    """
    negative = gdp_growth < 0
    # Récession = négatif pendant au moins 2 trimestres consécutifs
    recession = negative & negative.shift(1)
    return recession.fillna(False)


def get_recession_bands(gdp_growth: pd.Series) -> list:
    """
    Construit la liste des périodes de récession (début, fin).
    Utilisé pour ajouter des zones grises sur les graphiques.
    
    Retourne :
        Liste de tuples (date_debut, date_fin)
    """
    recessions = detect_recessions(gdp_growth)
    bands = []
    in_recession = False
    start = None
    
    for date, is_rec in recessions.items():
        if is_rec and not in_recession:
            start = date
            in_recession = True
        elif not is_rec and in_recession:
            bands.append((start, date))
            in_recession = False
    
    # Ferme une récession encore ouverte
    if in_recession:
        bands.append((start, recessions.index[-1]))
    
    return bands


def compute_hp_filter(series: pd.Series, lambda_param: float = 1600):
    """
    Filtre de Hodrick-Prescott pour décomposer une série en :
    - Tendance de long terme (cycle potentiel)
    - Composante cyclique (écart à la tendance = output gap approximatif)
    
    lambda = 1600 est le standard pour données trimestrielles (Hodrick & Prescott, 1997)
    
    Retourne :
        (cycle, trend) — deux pd.Series
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter
    
    # Supprime les NaN pour le filtre
    series_clean = series.dropna()
    cycle, trend = hpfilter(series_clean, lamb=lambda_param)
    
    return cycle, trend


def compute_output_gap(gdp_series: pd.Series) -> pd.Series:
    """
    Calcule l'output gap (écart de production) via le filtre HP.
    
    L'output gap = (PIB réel - PIB potentiel) / PIB potentiel × 100
    - Positif → économie en surchauffe (risque inflationniste)
    - Négatif → économie sous son potentiel (risque de déflation)
    
    Note : c'est une approximation. La méthode officielle de la 
    Commission Européenne utilise une fonction de production.
    """
    cycle, trend = compute_hp_filter(gdp_series)
    output_gap = (cycle / trend) * 100
    return output_gap


def create_analysis_dataset(data: dict) -> pd.DataFrame:
    """
    Crée un dataset unifié trimestriel pour la modélisation économétrique.
    Robuste aux changements de structure du DataFrame GDP.
    """
    gdp    = data["gdp"]
    unemp  = data["unemployment"]
    inf    = data["inflation"]
    rates  = data["rates"]

    # ── PIB : croissance trimestrielle ────────────────────────────
    # On cherche la colonne de croissance trimestrielle France
    if "France_growth_q" in gdp.columns:
        pib_growth = gdp["France_growth_q"].copy()
    elif "France" in gdp.columns:
        pib_growth = gdp["France"].pct_change() * 100
    else:
        pib_growth = pd.Series(dtype=float)

    # ── Chômage France ────────────────────────────────────────────
    chomage = unemp["France"].copy() if "France" in unemp.columns else pd.Series(dtype=float)

    # ── Inflation France ──────────────────────────────────────────
    inflation = inf["France"].copy() if "France" in inf.columns else pd.Series(dtype=float)

    # ── Taux OAT 10 ans ───────────────────────────────────────────
    oat_col = "OAT 10 ans (France)"
    oat = rates[oat_col].copy() if oat_col in rates.columns else pd.Series(dtype=float)

    # ── Taux BCE ──────────────────────────────────────────────────
    bce_col = "BCE (taux directeur)"
    bce = rates[bce_col].copy() if bce_col in rates.columns else pd.Series(dtype=float)

    # ── Rééchantillonnage trimestriel pour les séries mensuelles ──
    def to_quarterly(s):
        if s.empty:
            return s
        return s.resample("QS").mean()

    chomage_q   = to_quarterly(chomage)
    inflation_q = to_quarterly(inflation)
    oat_q       = to_quarterly(oat)
    bce_q       = to_quarterly(bce)

    # ── Assemblage ────────────────────────────────────────────────
    df = pd.DataFrame({
        "pib_growth": pib_growth,
        "chomage":    chomage_q,
        "inflation":  inflation_q,
        "taux_oat":   oat_q,
        "taux_bce":   bce_q,
    }).dropna(how="all")

    # Output gap si suffisamment d'observations
    if "France" in gdp.columns and len(gdp["France"].dropna()) > 20:
        df["output_gap"] = compute_output_gap(gdp["France"].dropna())

    return df
def compute_gdp_growth_annual_average(gdp_index: pd.Series) -> pd.Series:
    """
    Calcule la croissance annuelle du PIB par MOYENNE ANNUELLE.
    
    C'est la méthode officielle INSEE/Eurostat pour le chiffre annuel :
    moyenne des 4 trimestres de l'année N vs moyenne des 4 trimestres N-1.
    
    DIFFÉRENCE avec pct_change(4) :
    - pct_change(4) = glissement annuel trimestriel (T vs T-4 trimestres)
    - annual_average = variation de la moyenne annuelle (méthode comptabilité nationale)
    
    Les deux sont corrects, ils mesurent des choses légèrement différentes.
    Le dashboard affiche les DEUX pour transparence.
    """
    # Rééchantillonnage annuel par moyenne des 4 trimestres
    annual_mean = gdp_index.resample("YE").mean()
    return annual_mean.pct_change() * 100


def compute_gdp_growth_quarterly_yoy(gdp_index: pd.Series) -> pd.Series:
    """
    Glissement annuel trimestriel : variation T vs même trimestre T-1.
    Méthode standard pour les graphiques d'évolution conjoncturelle.
    """
    return gdp_index.pct_change(periods=4) * 100