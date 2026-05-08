# ============================================================
# src/models.py
# Modèles économétriques : tests de stationnarité, SARIMA, VAR
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools    import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api          import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import streamlit as st


# ══════════════════════════════════════════════════════════════
# SECTION 1 : Tests de stationnarité
# ══════════════════════════════════════════════════════════════

def test_adf(series: pd.Series, name: str = "") -> dict:
    """
    Test de Dickey-Fuller Augmenté (ADF).
    
    H0 : la série a une racine unitaire (non stationnaire)
    H1 : la série est stationnaire
    
    Si p-value < 0.05 → on rejette H0 → série stationnaire
    Si p-value > 0.05 → on ne rejette pas H0 → série non stationnaire
    
    On utilise la sélection automatique du nombre de retards
    via le critère BIC (plus parcimonieux que AIC sur petits échantillons).
    """
    series_clean = series.dropna()
    
    try:
        result = adfuller(
            series_clean,
            autolag="BIC",       # Sélection automatique des retards
            regression="ct",     # Constante + tendance (standard macro)
        )
        
        return {
            "nom":          name,
            "stat_adf":     round(result[0], 4),
            "p_value":      round(result[1], 4),
            "n_lags":       result[2],
            "n_obs":        result[3],
            "valeurs_crit": result[4],
            "stationnaire": result[1] < 0.05,
            "conclusion":   "I(0) stationnaire" if result[1] < 0.05 else "I(1) racine unitaire",
        }
    except Exception as e:
        return {"nom": name, "erreur": str(e), "stationnaire": None}


def test_kpss(series: pd.Series, name: str = "") -> dict:
    """
    Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin).
    
    COMPLÉMENTAIRE au test ADF — logique inversée :
    H0 : la série EST stationnaire (autour d'une tendance)
    H1 : la série a une racine unitaire
    
    Si p-value < 0.05 → on rejette H0 → série NON stationnaire
    
    Stratégie robuste : utiliser ADF ET KPSS ensemble.
    Si ADF rejette H0 ET KPSS ne rejette pas H0 → stationnarité confirmée.
    """
    series_clean = series.dropna()
    
    try:
        stat, p_value, n_lags, crit_values = kpss(
            series_clean,
            regression="ct",   # Cohérent avec ADF
            nlags="auto",
        )
        
        return {
            "nom":          name,
            "stat_kpss":    round(stat, 4),
            "p_value":      round(p_value, 4),
            "n_lags":       n_lags,
            "valeurs_crit": crit_values,
            "stationnaire": p_value >= 0.05,
            "conclusion":   "I(0) stationnaire" if p_value >= 0.05 else "I(1) racine unitaire",
        }
    except Exception as e:
        return {"nom": name, "erreur": str(e), "stationnaire": None}


def run_stationarity_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lance ADF + KPSS sur toutes les colonnes d'un DataFrame.
    Synthétise les résultats dans un tableau lisible.
    
    Retourne un DataFrame avec une ligne par variable.
    """
    rows = []
    
    for col in df.columns:
        serie = df[col].dropna()
        if len(serie) < 20:
            continue
        
        adf  = test_adf(serie, col)
        kpss_res = test_kpss(serie, col)
        
        # Diagnostic croisé ADF + KPSS
        adf_stat  = adf.get("stationnaire")
        kpss_stat = kpss_res.get("stationnaire")
        
        if adf_stat is True and kpss_stat is True:
            diagnostic = "✅ Stationnaire I(0)"
        elif adf_stat is False and kpss_stat is False:
            diagnostic = "❌ Racine unitaire I(1)"
        else:
            diagnostic = "⚠️ Résultats contradictoires"
        
        rows.append({
            "Variable":         col,
            "ADF stat":         adf.get("stat_adf", "N/A"),
            "ADF p-value":      adf.get("p_value", "N/A"),
            "ADF conclusion":   adf.get("conclusion", "N/A"),
            "KPSS stat":        kpss_res.get("stat_kpss", "N/A"),
            "KPSS p-value":     kpss_res.get("p_value", "N/A"),
            "KPSS conclusion":  kpss_res.get("conclusion", "N/A"),
            "Diagnostic final": diagnostic,
        })
    
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# SECTION 2 : Modèle SARIMA (prévisions univariées)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 4),
    forecast_horizon: int = 8,
    confidence_level: float = 0.90,
) -> dict:
    """
    Estime un modèle SARIMA et produit des prévisions.
    
    SARIMA(p,d,q)(P,D,Q,s) :
    - p : ordre AR (autorégression)
    - d : ordre d'intégration (différenciation)
    - q : ordre MA (moyenne mobile)
    - P, D, Q : composantes saisonnières
    - s : période saisonnière (4 = trimestriel)
    
    On utilise (1,1,1)(1,1,1,4) comme spécification de départ,
    robuste pour la plupart des séries macro trimestrielles.
    
    Arguments :
        series           : Série temporelle trimestrielle
        order            : (p, d, q)
        seasonal_order   : (P, D, Q, s)
        forecast_horizon : Nombre de trimestres à prévoir
        confidence_level : Niveau des intervalles de confiance
    
    Retourne :
        dict avec prévisions, intervalles, métriques
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 20:
        return {"erreur": "Série trop courte (minimum 20 observations)"}
    
    try:
        model = SARIMAX(
            series_clean,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c",
        )
        
        fitted = model.fit(disp=False, method="lbfgs")
        
        # Prévisions avec intervalles de confiance
        forecast = fitted.get_forecast(steps=forecast_horizon)
        forecast_mean = forecast.predicted_mean
        conf_int      = forecast.conf_int(alpha=1 - confidence_level)
        
        # Métriques in-sample
        fitted_values = fitted.fittedvalues
        residuals     = series_clean - fitted_values
        rmse = np.sqrt(np.mean(residuals**2))
        mae  = np.mean(np.abs(residuals))
        
        return {
            "modele":         "SARIMA",
            "ordre":          f"({order[0]},{order[1]},{order[2]})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})",
            "fitted_values":  fitted_values,
            "historique":     series_clean,
            "previsions":     forecast_mean,
            "ic_lower":       conf_int.iloc[:, 0],
            "ic_upper":       conf_int.iloc[:, 1],
            "aic":            round(fitted.aic, 2),
            "bic":            round(fitted.bic, 2),
            "rmse":           round(rmse, 4),
            "mae":            round(mae, 4),
            "niveau_confiance": confidence_level,
        }
    
    except Exception as e:
        # Fallback : ARIMA simple sans composante saisonnière
        try:
            model_simple = SARIMAX(
                series_clean,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                trend="c",
            )
            fitted = model_simple.fit(disp=False)
            forecast = fitted.get_forecast(steps=forecast_horizon)
            
            return {
                "modele":         "ARIMA (fallback)",
                "ordre":          "(1,1,1)",
                "fitted_values":  fitted.fittedvalues,
                "historique":     series_clean,
                "previsions":     forecast.predicted_mean,
                "ic_lower":       forecast.conf_int(alpha=1-confidence_level).iloc[:, 0],
                "ic_upper":       forecast.conf_int(alpha=1-confidence_level).iloc[:, 1],
                "aic":            round(fitted.aic, 2),
                "bic":            round(fitted.bic, 2),
                "rmse":           None,
                "mae":            None,
                "niveau_confiance": confidence_level,
                "note":           f"SARIMA échoué ({e}), fallback ARIMA(1,1,1)",
            }
        except Exception as e2:
            return {"erreur": f"Estimation impossible : {e2}"}
        
@st.cache_data(ttl=3600)
def fit_sarima_auto(
    series: pd.Series,
    forecast_horizon: int = 8,
    confidence_level: float = 0.90,
) -> dict:
    """
    SARIMA avec sélection automatique des ordres par grille AIC.
    Teste les combinaisons (p,d,q) ∈ {0,1,2}³ et retient le meilleur AIC.
    Méthode proche de l'algorithme auto.arima de R (Hyndman & Khandakar).
    """
    import itertools
    series_clean = series.dropna()

    if len(series_clean) < 20:
        return {"erreur": "Série trop courte"}

    best_aic   = np.inf
    best_model = None
    best_order = None
    best_sorder = None

    # Grille de recherche
    p_vals = range(0, 3)
    d_vals = range(0, 2)
    q_vals = range(0, 3)

    for p, d, q in itertools.product(p_vals, d_vals, q_vals):
        # On teste deux spécifications saisonnières
        for P, D, Q in [(1, 1, 1), (0, 0, 0)]:
            try:
                mod = SARIMAX(
                    series_clean,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 4),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="c",
                )
                res = mod.fit(disp=False, method="lbfgs")

                if res.aic < best_aic:
                    best_aic    = res.aic
                    best_model  = res
                    best_order  = (p, d, q)
                    best_sorder = (P, D, Q, 4)

            except Exception:
                continue

    if best_model is None:
        return {"erreur": "Aucun modèle convergé"}

    forecast   = best_model.get_forecast(steps=forecast_horizon)
    conf_int   = forecast.conf_int(alpha=1 - confidence_level)
    fitted_val = best_model.fittedvalues
    residuals  = series_clean - fitted_val
    rmse = np.sqrt(np.mean(residuals**2))
    mae  = np.mean(np.abs(residuals))

    return {
        "modele":           "SARIMA (sélection auto AIC)",
        "ordre":            f"({best_order[0]},{best_order[1]},{best_order[2]})({best_sorder[0]},{best_sorder[1]},{best_sorder[2]},4)",
        "fitted_values":    fitted_val,
        "historique":       series_clean,
        "previsions":       forecast.predicted_mean,
        "ic_lower":         conf_int.iloc[:, 0],
        "ic_upper":         conf_int.iloc[:, 1],
        "aic":              round(best_aic, 2),
        "bic":              round(best_model.bic, 2),
        "rmse":             round(rmse, 4),
        "mae":              round(mae, 4),
        "niveau_confiance": confidence_level,
    }        


# ══════════════════════════════════════════════════════════════
# SECTION 3 : Modèle VAR (prévisions multivariées)
# ══════════════════════════════════════════════════════════════

def select_var_lag(df_stationary: pd.DataFrame, max_lags: int = 8) -> dict:
    """
    Sélectionne le nombre optimal de retards pour le VAR.
    
    Critères utilisés (on retient le consensus) :
    - AIC  : Akaike Information Criterion (favorise ajustement)
    - BIC  : Bayesian Information Criterion (plus parcimonieux)
    - HQIC : Hannan-Quinn (compromis)
    - FPE  : Final Prediction Error
    
    Règle pratique : on prend max(AIC, BIC) comme compromis,
    avec un plafond à 4 pour données trimestrielles.
    """
    try:
        model = VAR(df_stationary.dropna())
        results = model.select_order(maxlags=max_lags)
        
        aic_lag  = results.aic
        bic_lag  = results.bic
        hqic_lag = results.hqic
        
        # Consensus : médiane des critères, plafonné à 4
        lag_consensus = int(np.median([aic_lag, bic_lag, hqic_lag]))
        lag_final     = min(max(lag_consensus, 1), 4)
        
        return {
            "aic_lag":  aic_lag,
            "bic_lag":  bic_lag,
            "hqic_lag": hqic_lag,
            "lag_retenu": lag_final,
            "tableau":    results.summary(),
        }
    except Exception as e:
        return {"lag_retenu": 2, "erreur": str(e)}


@st.cache_data(ttl=3600)
def fit_var(
    df: pd.DataFrame,
    n_lags: int = 2,
    forecast_horizon: int = 8,
    confidence_level: float = 0.90,
) -> dict:
    """
    Estime un modèle VAR(p) sur données STATIONNAIRES.
    
    IMPORTANT : Le VAR requiert des séries stationnaires.
    On différencie donc les séries I(1) avant estimation,
    puis on re-cumule les prévisions pour revenir au niveau.
    
    Variables incluses (dataset analyse France) :
    - Croissance PIB trimestrielle (déjà stationnaire)
    - Taux de chômage (différencié si I(1))
    - Inflation (généralement stationnaire)
    - Taux d'intérêt (différencié si I(1))
    
    Retourne :
        dict avec prévisions VAR, intervalles, IRF
    """
    try:
        # Nettoyage : supprime les colonnes trop lacunaires
        df_clean = df.dropna()
        
        if len(df_clean) < 4 * n_lags + 10:
            return {
                "erreur": f"Échantillon trop petit ({len(df_clean)} obs) "
                          f"pour VAR({n_lags})"
            }
        
        # Estimation du VAR
        model  = VAR(df_clean)
        fitted = model.fit(n_lags)
        
        # Prévisions
        # Le VAR prédit à partir des dernières n_lags observations
        last_obs = df_clean.values[-n_lags:]
        forecast_result = fitted.forecast_interval(
            y=last_obs,
            steps=forecast_horizon,
            alpha=1 - confidence_level,
        )
        
        forecast_mean  = forecast_result[0]
        forecast_lower = forecast_result[1]
        forecast_upper = forecast_result[2]
        
        # Création de l'index futur (trimestres)
        last_date   = df_clean.index[-1]
        future_idx  = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=forecast_horizon,
            freq="QS",
        )
        
        # Conversion en DataFrames
        df_forecast = pd.DataFrame(
            forecast_mean,
            index=future_idx,
            columns=df_clean.columns,
        )
        df_lower = pd.DataFrame(
            forecast_lower,
            index=future_idx,
            columns=df_clean.columns,
        )
        df_upper = pd.DataFrame(
            forecast_upper,
            index=future_idx,
            columns=df_clean.columns,
        )
        
        # Métriques du modèle
        return {
            "modele":           "VAR",
            "n_lags":           n_lags,
            "n_variables":      len(df_clean.columns),
            "n_obs":            len(df_clean),
            "variables":        list(df_clean.columns),
            "historique":       df_clean,
            "fitted":           fitted,
            "previsions":       df_forecast,
            "ic_lower":         df_lower,
            "ic_upper":         df_upper,
            "aic":              round(fitted.aic, 2),
            "bic":              round(fitted.bic, 2),
            "niveau_confiance": confidence_level,
        }
    
    except Exception as e:
        return {"erreur": f"Estimation VAR impossible : {e}"}


# ══════════════════════════════════════════════════════════════
# SECTION 4 : Test de cointégration de Johansen
# ══════════════════════════════════════════════════════════════

def test_johansen(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """
    Test de cointégration de Johansen.
    
    Détecte si des séries I(1) partagent une relation de long terme
    (vecteur cointégrant), auquel cas un VECM serait plus approprié
    qu'un VAR en différences.
    
    det_order :
        -1 = pas de constante
         0 = constante dans la relation de cointégration (standard)
         1 = constante + tendance
    
    Retourne le nombre de relations de cointégration détectées
    au seuil de 5%.
    """
    try:
        df_clean = df.dropna()
        result   = coint_johansen(df_clean, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Statistiques trace
        trace_stat = result.lr1        # Statistiques du test trace
        crit_vals  = result.cvt        # Valeurs critiques (90%, 95%, 99%)
        
        # Nombre de relations de cointégration (seuil 5% = colonne 1)
        n_coint = int(np.sum(trace_stat > crit_vals[:, 1]))
        
        rows = []
        for i in range(len(trace_stat)):
            rows.append({
                "H0 (rang ≤)":       i,
                "Stat. trace":       round(trace_stat[i], 3),
                "Valeur critique 5%":round(crit_vals[i, 1], 3),
                "Rejet H0 (5%)":     "✅ Oui" if trace_stat[i] > crit_vals[i, 1] else "❌ Non",
            })
        
        return {
            "n_coint":      n_coint,
            "tableau":      pd.DataFrame(rows),
            "conclusion":   (
                f"{n_coint} relation(s) de cointégration détectée(s). "
                + ("→ VECM recommandé." if n_coint > 0 else "→ VAR en différences approprié.")
            ),
            "vecteurs":     result.evec,
        }
    
    except Exception as e:
        return {"erreur": str(e), "n_coint": 0}

# ══════════════════════════════════════════════════════════════
# SECTION 5 : Test de Zivot-Andrews (ruptures structurelles)
# ══════════════════════════════════════════════════════════════

def test_zivot_andrews(series: pd.Series, name: str = "") -> dict:
    """
    Test de Zivot-Andrews (1992) — extension du test ADF avec
    rupture structurelle endogène.
    
    Motivation : le test ADF standard est biaisé vers la non-rejet
    de H0 (racine unitaire) en présence de ruptures structurelles
    (ex: crise 2008, Covid). Zivot-Andrews détecte ENDOGÈNEMENT
    la date de rupture la plus probable.
    
    H0 : racine unitaire AVEC rupture structurelle
    H1 : stationnarité avec une rupture de niveau ou de tendance
    
    model='both' : rupture sur constante ET tendance (le plus général)
    """
    from statsmodels.tsa.stattools import zivot_andrews
    
    series_clean = series.dropna()
    
    if len(series_clean) < 30:
        return {"nom": name, "erreur": "Série trop courte (min 30 obs)"}
    
    try:
        za_stat, za_pval, za_cvdict, za_bpidx, za_baselag = zivot_andrews(
            series_clean,
            trim=0.15,       # Exclut 15% de chaque extrémité
            maxlag=4,
            regression="ct", # Constante + tendance
            autolag="AIC",
        )
        
        # Date de rupture détectée
        breakpoint_date = series_clean.index[za_bpidx]
        
        # Valeurs critiques standard ZA
        cv_1pct  = za_cvdict.get("1%",  -5.57)
        cv_5pct  = za_cvdict.get("5%",  -5.08)
        cv_10pct = za_cvdict.get("10%", -4.82)
        
        stationnaire = za_stat < cv_5pct  # Seuil 5%
        
        return {
            "nom":              name,
            "stat_za":          round(za_stat, 4),
            "p_value":          round(za_pval, 4),
            "breakpoint_date":  breakpoint_date,
            "breakpoint_idx":   za_bpidx,
            "n_lags":           za_baselag,
            "cv_1pct":          cv_1pct,
            "cv_5pct":          cv_5pct,
            "cv_10pct":         cv_10pct,
            "stationnaire":     stationnaire,
            "conclusion": (
                f"Stationnaire avec rupture en {str(breakpoint_date)[:10]}"
                if stationnaire else
                f"Racine unitaire (rupture détectée en {str(breakpoint_date)[:10]})"
            ),
        }
    
    except Exception as e:
        return {"nom": name, "erreur": str(e), "stationnaire": None}


def run_full_stationarity_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batterie complète : ADF + KPSS + Zivot-Andrews.
    Version enrichie de run_stationarity_battery pour la page Économétrie.
    """
    rows = []
    
    for col in df.columns:
        serie = df[col].dropna()
        if len(serie) < 30:
            continue
        
        adf_res = test_adf(serie, col)
        kpss_res = test_kpss(serie, col)
        za_res   = test_zivot_andrews(serie, col)
        
        # Diagnostic croisé
        adf_stat  = adf_res.get("stationnaire")
        kpss_stat = kpss_res.get("stationnaire")
        za_stat   = za_res.get("stationnaire")
        
        # Règle de décision enrichie
        votes_stat = sum([
            1 if adf_stat  is True else 0,
            1 if kpss_stat is True else 0,
            1 if za_stat   is True else 0,
        ])
        
        if votes_stat >= 2:
            diagnostic = "✅ Stationnaire I(0)"
            ordre_integ = "I(0)"
        else:
            diagnostic = "❌ Racine unitaire I(1)"
            ordre_integ = "I(1)"
        
        rows.append({
            "Variable":          col,
            "ADF stat":          adf_res.get("stat_adf", "N/A"),
            "ADF p-val":         adf_res.get("p_value", "N/A"),
            "KPSS stat":         kpss_res.get("stat_kpss", "N/A"),
            "KPSS p-val":        kpss_res.get("p_value", "N/A"),
            "ZA stat":           za_res.get("stat_za", "N/A"),
            "ZA rupture":        str(za_res.get("breakpoint_date", "N/A"))[:10],
            "Ordre intégration": ordre_integ,
            "Diagnostic":        diagnostic,
        })
    
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# SECTION 6 : Modèle VECM
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fit_vecm(
    df: pd.DataFrame,
    n_coint: int = 1,
    k_ar_diff: int = 1,
    forecast_horizon: int = 8,
    confidence_level: float = 0.90,
) -> dict:
    """
    Estime un modèle VECM (Vector Error Correction Model).
    
    Le VECM est approprié quand :
    1. Les séries sont I(1) (racines unitaires)
    2. Elles sont cointégrées (relation de long terme commune)
    
    Structure du VECM :
    ΔYt = αβ'Yt-1 + Γ1ΔYt-1 + ... + Γk-1ΔYt-k+1 + εt
    
    où :
    - β  = vecteur(s) cointégrant(s) (relation de long terme)
    - α  = vitesses d'ajustement vers l'équilibre
    - Γi = matrices de dynamique de court terme
    
    Avantage vs VAR en différences : exploite l'information
    de long terme contenue dans les niveaux des séries.
    """
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    df_clean = df.dropna()
    
    if len(df_clean) < 30:
        return {"erreur": "Échantillon insuffisant pour le VECM"}
    
    try:
        model = VECM(
            df_clean,
            k_ar_diff=k_ar_diff,   # Retards en différences
            coint_rank=n_coint,    # Nombre de relations cointégrantes
            deterministic="co",    # Constante dans la relation de coint.
        )
        fitted = model.fit()
        
        # Prévisions VECM
        # statsmodels 0.14 : predict() retourne uniquement les moyennes
        forecast_arr = fitted.predict(steps=forecast_horizon)
        
        # Index futur
        last_date  = df_clean.index[-1]
        future_idx = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=forecast_horizon,
            freq="QS",
        )
        
        df_forecast = pd.DataFrame(
            forecast_arr,
            index=future_idx,
            columns=df_clean.columns,
        )
        
        return {
            "modele":        "VECM",
            "n_coint":       n_coint,
            "k_ar_diff":     k_ar_diff,
            "n_obs":         len(df_clean),
            "variables":     list(df_clean.columns),
            "historique":    df_clean,
            "fitted":        fitted,
            "previsions":    df_forecast,
            "alpha":         pd.DataFrame(    # Vitesses d'ajustement
                fitted.alpha,
                index=df_clean.columns,
                columns=[f"EC{i+1}" for i in range(n_coint)],
            ),
            "beta":          pd.DataFrame(    # Vecteurs cointégrants
                fitted.beta,
                index=df_clean.columns,
                columns=[f"CV{i+1}" for i in range(n_coint)],
            ),
        }
    
    except Exception as e:
        return {"erreur": f"Estimation VECM impossible : {e}"}


# ══════════════════════════════════════════════════════════════
# SECTION 7 : Fonctions de réponse impulsionnelle (IRF)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def compute_irf(
    _var_fitted,              # Modèle VAR estimé (underscore = non hashable)
    impulse: str,             # Variable choc
    response: str,            # Variable réponse
    periods: int = 12,
    confidence_level: float = 0.90,
) -> dict:
    """
    Calcule la Fonction de Réponse Impulsionnelle (IRF).
    
    L'IRF mesure comment un choc unitaire (1 écart-type) sur
    la variable 'impulse' se propage sur 'response' au fil du temps.
    
    Identification : décomposition de Cholesky (ordering matter !)
    Ordre retenu : PIB → Inflation → Chômage
    (hypothèse : le PIB réagit en premier aux chocs d'offre,
    l'inflation ajuste ensuite, le chômage en dernier = courbe de Phillips)
    
    Limite : les résultats dépendent de l'ordre de Cholesky choisi.
    Une identification structurelle rigoureuse nécessiterait des
    restrictions théoriques additionnelles (SVAR).
    """
    try:
        alpha = 1 - confidence_level
        irf   = _var_fitted.irf(periods=periods)
        
        # Indices des variables
        variables  = list(_var_fitted.names)
        imp_idx    = variables.index(impulse)
        resp_idx   = variables.index(response)
        
        # Réponse ponctuelle
        irf_vals   = irf.orth_irfs[:, resp_idx, imp_idx]
        
        # Intervalles de confiance bootstrap
        # statsmodels 0.14 : méthode errband_mc (sans underscore)
        try:
            irf_ci    = irf.errband_mc(
                orth=True,
                repl=200,
                signif=alpha,
            )
            irf_lower = irf_ci[0][:, resp_idx, imp_idx]
            irf_upper = irf_ci[1][:, resp_idx, imp_idx]
        except Exception:
            # Fallback : intervalles asymptotiques si bootstrap échoue
            irf_lower = irf_vals - 1.645 * np.std(irf_vals)
            irf_upper = irf_vals + 1.645 * np.std(irf_vals)
        
        return {
            "impulse":   impulse,
            "response":  response,
            "periodes":  list(range(periods + 1)),
            "irf":       irf_vals,
            "ic_lower":  irf_lower,
            "ic_upper":  irf_upper,
        }
    
    except Exception as e:
        return {"erreur": str(e)}


# ══════════════════════════════════════════════════════════════
# SECTION 8 : Scénarios "what-if"
# ══════════════════════════════════════════════════════════════

def compute_scenario(
    base_forecast: pd.DataFrame,
    variable: str,
    choc_amplitude: float,
    choc_persistance: int,
    propagation: dict = None,
) -> pd.DataFrame:
    """
    Applique un choc exogène sur les prévisions de base.
    
    Méthode : choc dégressif de type AR(1) sur la variable cible,
    avec propagation calibrée vers les autres variables selon
    des multiplicateurs issus de la littérature empirique.
    
    Multiplicateurs de référence (FMI/BCE) :
    - Choc taux +100pb → PIB −0.2% à 1 an, −0.4% à 2 ans
    - Choc pétrole +10% → inflation +0.3pp, PIB −0.1%
    - Choc demande +1% PIB → chômage −0.3pp (loi d'Okun)
    
    Arguments :
        base_forecast   : DataFrame des prévisions baseline
        variable        : Variable impactée en premier
        choc_amplitude  : Amplitude du choc (unité de la variable)
        choc_persistance: Nombre de trimestres de persistance
        propagation     : Dict {var: multiplicateur} pour les effets croisés
    
    Retourne :
        DataFrame avec prévisions sous scénario de choc
    """
    scenario = base_forecast.copy()
    
    if variable not in scenario.columns:
        return scenario
    
    # Choc dégressif : amplitude × ρ^t avec ρ = 0.7 (persistance AR)
    rho = 0.7
    n_periods = min(choc_persistance, len(scenario))
    
    for t in range(n_periods):
        choc_t = choc_amplitude * (rho ** t)
        scenario.iloc[t][variable] += choc_t
        
        # Propagation vers les autres variables
        if propagation:
            for var_cible, multiplicateur in propagation.items():
                if var_cible in scenario.columns:
                    scenario.iloc[t][var_cible] += choc_t * multiplicateur
    
    return scenario
    
