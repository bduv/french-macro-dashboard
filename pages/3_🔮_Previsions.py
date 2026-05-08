# ============================================================
# pages/3_🔮_Previsions.py
# Prévisions macroéconomiques : SARIMA + VAR
# Intervalles de confiance + scénarios
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader    import load_all_data
from src.data_processor import create_analysis_dataset
from src.models         import (
    fit_sarima, fit_var, select_var_lag,
    run_stationarity_battery, test_johansen,
)
from src.utils          import apply_house_style
from config.settings    import COLORS, MODEL_CONFIG

from src.models import (
    fit_sarima, fit_sarima_auto, fit_var, select_var_lag,
    run_stationarity_battery, test_johansen,
)

st.set_page_config(
    page_title="Prévisions · French Macro Dashboard",
    page_icon="🔮",
    layout="wide",
)

def load_css():
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "style.css"
    )
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🔮 Prévisions Macroéconomiques")
st.markdown("*Modèles SARIMA & VAR · Intervalles de confiance à 90%*")
st.markdown("---")

# ── Chargement & préparation des données ──────────────────────
with st.spinner("Chargement des données..."):
    data    = load_all_data()
    df_ana  = create_analysis_dataset(data)

# ── Sidebar : paramètres des modèles ──────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    horizon = st.slider(
        "Horizon de prévision (trimestres)",
        min_value=2, max_value=12, value=8, step=1,
    )
    conf_level = st.select_slider(
        "Intervalle de confiance",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x*100)}%",
    )
    variable_sarima = st.selectbox(
        "Variable à prévoir (SARIMA)",
        options=["pib_growth", "chomage", "inflation", "taux_oat"],
        format_func=lambda x: {
            "pib_growth": "Croissance PIB",
            "chomage":    "Chômage",
            "inflation":  "Inflation IPCH",
            "taux_oat":   "OAT 10 ans",
        }.get(x, x),
    )
    n_lags_var = st.slider(
        "Retards VAR",
        min_value=1, max_value=4, value=2,
    )

    st.markdown("---")
    st.markdown("**Note méthodologique**")
    st.caption(
        "Les prévisions SARIMA sont univariées (une variable à la fois). "
        "Le VAR est multivarié et capture les interdépendances. "
        "Les intervalles sont calculés analytiquement (SARIMA) "
        "et par simulation (VAR)."
    )

# ══════════════════════════════════════════════════════════════
# SECTION 1 : Prévisions SARIMA
# ══════════════════════════════════════════════════════════════

st.subheader("📈 Prévisions SARIMA — Modèle univarié")

labels = {
    "pib_growth": ("Croissance PIB trimestrielle (%)", COLORS["France"]),
    "chomage":    ("Taux de chômage (%)", "#e74c3c"),
    "inflation":  ("Inflation IPCH (%)", "#f39c12"),
    "taux_oat":   ("OAT 10 ans (%)", "#8e44ad"),
}

if variable_sarima in df_ana.columns:
    serie_cible = df_ana[variable_sarima].dropna()
    label_fr, couleur = labels[variable_sarima]

    with st.spinner(f"Sélection automatique du meilleur SARIMA pour {label_fr}..."):
        result_sarima = fit_sarima_auto(
            series=serie_cible,
            forecast_horizon=horizon,
            confidence_level=conf_level,
        )

    if "erreur" not in result_sarima:
        # ── Graphique de prévision ────────────────────────────
        fig_sarima = go.Figure()

        # ── 1. IC EN PREMIER (en dessous de tout) ─────────────────
        ic_up  = result_sarima["ic_upper"]
        ic_low = result_sarima["ic_lower"]

        fig_sarima.add_trace(go.Scatter(
            x=list(ic_up.index) + list(ic_low.index[::-1]),
            y=list(ic_up.values) + list(ic_low.values[::-1]),
            fill="toself",
            fillcolor="rgba(0, 35, 149, 0.10)",  # bleu très transparent
            line=dict(color="rgba(0,0,0,0)"),
            name=f"IC {int(conf_level*100)}%",
            showlegend=True,
            hoverinfo="skip",
        ))

        # ── 2. Historique ─────────────────────────────────────────
        hist = result_sarima["historique"].iloc[-40:]
        fig_sarima.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            name="Historique",
            line=dict(color=couleur, width=2.5),
        ))

        # ── 3. Valeurs ajustées ───────────────────────────────────
        fitted_plot = result_sarima["fitted_values"].iloc[-40:]
        fig_sarima.add_trace(go.Scatter(
            x=fitted_plot.index, y=fitted_plot.values,
            name="Valeurs ajustées",
            line=dict(color=couleur, width=1.5, dash="dot"),
            opacity=0.5,
        ))

        # ── 4. Prévisions EN DERNIER (au-dessus de tout) ──────────
        prev = result_sarima["previsions"]   # ← AJOUTE CETTE LIGNE
        fig_sarima.add_trace(go.Scatter(
            x=prev.index, y=prev.values,
            name=f"Prévision ({horizon} trim.)",
            line=dict(color="#e74c3c", width=3),  # rouge vif, bien visible
        ))

        # ── 5. Marqueur de séparation ─────────────────────────────
        fig_sarima.add_vrect(
            x0=hist.index[-1],
            x1=hist.index[-1] + pd.DateOffset(days=1),
            line_width=1.5,
            line_color="#999999",
            line_dash="dash",
        )
        fig_sarima.add_annotation(
            x=hist.index[-1],
            y=1.02, yref="paper",
            text="→ Prévision",
            showarrow=False,
            font=dict(size=11, color="#666666"),
        )

        fig_sarima = apply_house_style(
            fig_sarima,
            title=f"Prévision SARIMA{result_sarima['ordre']} — {label_fr}",
            height=480,
        )
        st.plotly_chart(fig_sarima, use_container_width=True)

        # ── Métriques du modèle ───────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modèle", result_sarima["ordre"])
        with col2:
            st.metric("AIC", result_sarima["aic"])
        with col3:
            st.metric("BIC", result_sarima["bic"])
        with col4:
            rmse = result_sarima.get("rmse")
            st.metric("RMSE", f"{rmse:.4f}" if rmse else "N/A")

        # ── Tableau des prévisions ────────────────────────────
        with st.expander("📋 Tableau des prévisions détaillées"):
            df_prev_table = pd.DataFrame({
                "Trimestre":          prev.index.strftime("%Y-T%q"),
                "Prévision":          prev.values.round(3),
                f"IC {int(conf_level*100)}% — Borne inf.": ic_low.values.round(3),
                f"IC {int(conf_level*100)}% — Borne sup.": ic_up.values.round(3),
            })
            st.dataframe(df_prev_table, use_container_width=True, hide_index=True)

        if "note" in result_sarima:
            st.info(f"ℹ️ {result_sarima['note']}")

    else:
        st.error(f"❌ {result_sarima['erreur']}")

st.markdown("""
<div class="insight-box">
<b>📌 SARIMA :</b> Le modèle SARIMA (Seasonal AutoRegressive Integrated Moving Average)
est le standard pour les prévisions de séries temporelles univariées en macroéconomie.
Il capture les dynamiques autorégressives, la tendance (différenciation) et la saisonnalité.
Sa limite principale est de n'exploiter qu'une seule variable — c'est pourquoi on le
complète avec un VAR multivarié ci-dessous.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2 : Prévisions VAR
# ══════════════════════════════════════════════════════════════

st.subheader("🔗 Prévisions VAR — Modèle multivarié")

st.markdown("""
Le VAR (Vector AutoRegression) prédit simultanément plusieurs variables en
capturant leurs **interdépendances dynamiques**. Chaque variable est expliquée
par ses propres valeurs passées ET par les valeurs passées des autres variables.
""")

# Variables pour le VAR — on utilise uniquement les colonnes disponibles
# Variables pour le VAR
var_cols_ideales = ["pib_growth", "chomage", "inflation"]
var_cols = [c for c in var_cols_ideales if c in df_ana.columns
            and df_ana[c].dropna().shape[0] > 20]

if len(var_cols) >= 2:

    df_var_raw = df_ana[var_cols].dropna()

    # Différenciation des séries non stationnaires
    # pib_growth : déjà un taux de variation → stationnaire
    # chomage, inflation : niveaux I(1) → on différencie
    df_var = pd.DataFrame(index=df_var_raw.index)

    series_a_differencier = ["chomage", "inflation", "taux_oat"]

    for col in var_cols:
        if col in series_a_differencier:
            df_var[f"d_{col}"] = df_var_raw[col].diff()
        else:
            df_var[col] = df_var_raw[col]

    df_var = df_var.dropna()
    df_var = df_var.dropna()
    
    # ── IMPORTANT : mettre à jour var_cols avec les vraies colonnes ──
    # Après différenciation, les noms ont changé (chomage → d_chomage)
    var_cols = list(df_var.columns)

    # Labels mis à jour pour l'affichage
    labels_var = {
        "pib_growth":  "Croissance PIB (%)",
        "d_chomage":   "Δ Chômage (pp)",
        "d_inflation": "Δ Inflation (pp)",
        "d_taux_oat":  "Δ OAT 10 ans (pp)",
        "chomage":     "Chômage (%)",
        "inflation":   "Inflation (%)",
    }

    st.info(
        "ℹ️ **Note méthodologique** : le chômage et l'inflation étant des séries I(1) "
        "(racine unitaire confirmée par ADF+KPSS), le VAR est estimé sur leurs "
        "**différences premières** (Δ). Les prévisions représentent donc des variations, "
        "non des niveaux. Pour travailler en niveaux avec des séries I(1) cointégrées, "
        "un VECM sera estimé au Jour 4."
    )

    with st.spinner("Estimation du modèle VAR..."):
        result_var = fit_var(
            df=df_var,
            n_lags=n_lags_var,
            forecast_horizon=horizon,
            confidence_level=conf_level,
        )

    if "erreur" not in result_var:

        # ── Graphiques VAR : une colonne par variable ─────────
        cols_plot = st.columns(len(var_cols))

        couleurs_var = {
            "pib_growth": COLORS["France"],
            "chomage":    "#e74c3c",
            "inflation":  "#f39c12",
            "taux_oat":   "#8e44ad",
        }
        labels_var = {
            "pib_growth": "Croissance PIB (%)",
            "chomage":    "Chômage (%)",
            "inflation":  "Inflation (%)",
            "taux_oat":   "OAT 10 ans (%)",
        }

        for i, col_var in enumerate(var_cols):
            with cols_plot[i]:
                hist_var = result_var["historique"][col_var].iloc[-32:]
                prev_var = result_var["previsions"][col_var]
                ic_l_var = result_var["ic_lower"][col_var]
                ic_u_var = result_var["ic_upper"][col_var]
                coul_var = couleurs_var.get(col_var, "#002395")

                fig_v = go.Figure()

                # 1. IC en premier
                fig_v.add_trace(go.Scatter(
                    x=list(ic_u_var.index) + list(ic_l_var.index[::-1]),
                    y=list(ic_u_var.values) + list(ic_l_var.values[::-1]),
                    fill="toself",
                    fillcolor=f"rgba{tuple(list(bytes.fromhex(coul_var.lstrip('#'))) + [20])}",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"IC {int(conf_level*100)}%",
                    hoverinfo="skip",
                    showlegend=False,
                ))

                # 2. Historique
                fig_v.add_trace(go.Scatter(
                    x=hist_var.index, y=hist_var.values,
                    name="Historique",
                    line=dict(color=coul_var, width=2),
                ))

                # 3. Prévision en dernier
                fig_v.add_trace(go.Scatter(
                    x=prev_var.index, y=prev_var.values,
                    name="Prévision",
                    line=dict(color="#e74c3c", width=2.5),
                ))

                fig_v.add_vrect(
                    x0=hist_var.index[-1],
                    x1=hist_var.index[-1] + pd.DateOffset(days=1),
                    line_width=1,
                    line_color="#aaaaaa",
                    line_dash="dash",
                )

                fig_v = apply_house_style(
                    fig_v,
                    title=labels_var.get(col_var, col_var),
                    height=320,
                )
                fig_v.update_layout(showlegend=False)
                st.plotly_chart(fig_v, use_container_width=True)

        # ── Métriques VAR ─────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Modèle", f"VAR({result_var['n_lags']})")
        with c2: st.metric("Variables", result_var["n_variables"])
        with c3: st.metric("AIC", result_var["aic"])
        with c4: st.metric("Observations", result_var["n_obs"])

        # ── Tableau des prévisions VAR ────────────────────────
        with st.expander("📋 Tableau complet des prévisions VAR"):
            df_var_table = result_var["previsions"].round(3).copy()
            df_var_table.index = df_var_table.index.strftime("%Y-T%q")
            df_var_table.columns = [labels_var.get(c, c) for c in df_var_table.columns]
            st.dataframe(df_var_table, use_container_width=True)

    else:
        st.error(f"❌ {result_var['erreur']}")

else:
    st.warning("Données insuffisantes pour le VAR (minimum 2 variables requises).")

st.markdown("""
<div class="insight-box">
<b>📌 VAR :</b> Le modèle VAR capture les <i>effets de propagation</i> entre variables :
un choc sur l'inflation affecte les anticipations de taux, qui influencent l'investissement,
qui impacte la croissance et l'emploi. Cette structure est celle utilisée par les grandes
banques centrales pour leurs projections macroéconomiques.
<b>Limite :</b> le VAR est un modèle réduit (forme réduite) — il décrit les corrélations
historiques sans les identifier causalement. Pour l'identification structurelle, il faudrait
un SVAR avec restrictions théoriques (Jour 4).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3 : Tests de stationnarité (résumé)
# ══════════════════════════════════════════════════════════════

st.subheader("🔬 Tests de stationnarité (ADF + KPSS)")

with st.expander("Voir les résultats des tests — prérequis pour la modélisation"):
    st.markdown("""
    La stationnarité est un **prérequis fondamental** pour les modèles VAR.
    Une série est stationnaire si sa moyenne et sa variance sont constantes dans le temps.
    On combine ADF et KPSS pour un diagnostic robuste.
    """)

    with st.spinner("Tests en cours..."):
        # Les tests se font sur les colonnes ORIGINALES de df_ana (avant différenciation)
        cols_originales = ["pib_growth", "chomage", "inflation"]
        cols_dispo = [c for c in cols_originales if c in df_ana.columns]
        df_tests = run_stationarity_battery(df_ana[cols_dispo].dropna())

    st.dataframe(df_tests, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="insight-box">
<b>📌 Pourquoi des contradictions ADF/KPSS ?</b><br>
Les contradictions entre ADF et KPSS sont <b>courantes et attendues</b> en macroéconomie.
Elles surviennent pour trois raisons principales :<br>
<b>(1) Puissance faible</b> : avec 30-40 ans de données trimestrielles (~120-140 obs.),
les deux tests manquent de puissance statistique — leurs conclusions sont sensibles
aux points de rupture structurels (crise 2008, Covid).<br>
<b>(2) Hypothèses inversées</b> : ADF teste H₀ = racine unitaire, KPSS teste
H₀ = stationnarité. Un rejet simultané est logiquement possible si la série
est à la frontière I(0)/I(1).<br>
<b>(3) Ruptures structurelles</b> : une série stationnaire par morceaux peut
paraître I(1) au test ADF. La solution rigoureuse serait un test de Zivot-Andrews
(rupture endogène), implémenté au Jour 4.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    f"🇫🇷 French Macro Forecast Dashboard · "
    f"Modèles : SARIMA, VAR · "
    f"Dernière mise à jour : {data['last_update']}"
)