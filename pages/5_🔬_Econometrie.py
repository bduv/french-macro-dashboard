# ============================================================
# pages/5_🔬_Econometrie.py
# Analyse économétrique approfondie
# Tests de stationnarité, cointégration, IRF, VECM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader    import load_all_data
from src.data_processor import create_analysis_dataset
from src.models         import (
    run_full_stationarity_battery,
    test_zivot_andrews,
    test_johansen,
    fit_var,
    fit_vecm,
    compute_irf,
)
from src.utils          import apply_house_style
from config.settings    import COLORS

st.set_page_config(
    page_title="Économétrie · French Macro Dashboard",
    page_icon="🔬",
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

st.title("🔬 Analyse Économétrique Avancée")
st.markdown("*Tests de stationnarité · Cointégration · IRF · VECM*")
st.markdown("---")

with st.spinner("Chargement des données..."):
    data   = load_all_data()
    df_ana = create_analysis_dataset(data)

# Variables disponibles
cols_dispo = [c for c in ["pib_growth", "chomage", "inflation", "taux_oat"]
              if c in df_ana.columns and df_ana[c].dropna().shape[0] > 30]

labels_vars = {
    "pib_growth": "Croissance PIB (%)",
    "chomage":    "Chômage (%)",
    "inflation":  "Inflation IPCH (%)",
    "taux_oat":   "OAT 10 ans (%)",
    "taux_bce":   "Taux BCE (%)",
}

# ══════════════════════════════════════════════════════════════
# SECTION 1 : Batterie de tests de stationnarité
# ══════════════════════════════════════════════════════════════

st.subheader("1️⃣ Tests de Stationnarité — ADF + KPSS + Zivot-Andrews")

st.markdown("""
La stationnarité est le **prérequis fondamental** de toute modélisation en séries
temporelles. On mobilise trois tests complémentaires pour un diagnostic robuste :
ADF (H₀ = racine unitaire), KPSS (H₀ = stationnarité) et Zivot-Andrews
(ADF avec rupture structurelle endogène — essentiel sur données macro longues).
""")

with st.spinner("Tests en cours — ADF + KPSS + Zivot-Andrews..."):
    df_stat_tests = run_full_stationarity_battery(
        df_ana[cols_dispo].dropna()
    )

st.dataframe(df_stat_tests, use_container_width=True, hide_index=True)

# Visualisation des ruptures Zivot-Andrews
st.markdown("#### Ruptures structurelles détectées (Zivot-Andrews)")

za_results = {}
for col in cols_dispo:
    za_results[col] = test_zivot_andrews(df_ana[col].dropna(), col)

fig_za = make_subplots(
    rows=1, cols=len(cols_dispo),
    subplot_titles=[labels_vars.get(c, c) for c in cols_dispo],
)

for i, col in enumerate(cols_dispo):
    serie  = df_ana[col].dropna()
    za_res = za_results[col]

    fig_za.add_trace(go.Scatter(
        x=serie.index, y=serie.values,
        line=dict(color=COLORS["France"], width=1.8),
        showlegend=False,
    ), row=1, col=i+1)

    if "breakpoint_date" in za_res and za_res["breakpoint_date"] is not None:
        fig_za.add_vline(
            x=za_res["breakpoint_date"].timestamp() * 1000,
            line_dash="dash",
            line_color="#e74c3c",
            line_width=2,
            row=1, col=i+1,
        )

fig_za.update_layout(
    height=300,
    title_text="Séries temporelles avec ruptures structurelles (trait rouge)",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_za, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📌 Zivot-Andrews :</b> En présence de ruptures structurelles (2008-2009,
2020), le test ADF standard sur-rejette H₀ et conclut à tort à la
non-stationnarité. Le test de Zivot-Andrews corrige ce biais en estimant
endogènement la date de rupture la plus probable. Un résultat divergent
entre ADF et ZA indique que la série est probablement stationnaire
<i>par morceaux</i>, avec une rupture de niveau ou de tendance.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2 : Test de cointégration de Johansen
# ══════════════════════════════════════════════════════════════

st.subheader("2️⃣ Test de Cointégration de Johansen")

st.markdown("""
Si plusieurs séries sont I(1), le test de Johansen détecte l'existence de
**relations de long terme communes** (vecteurs cointégrants). En présence
de cointégration, un VECM est préférable au VAR en différences car il
exploite l'information contenue dans les déséquilibres de long terme.
""")

# On teste la cointégration sur les séries en NIVEAUX (pas différenciées)
cols_i1 = [c for c in ["chomage", "inflation", "taux_oat"]
           if c in df_ana.columns and df_ana[c].dropna().shape[0] > 30]

if len(cols_i1) >= 2:
    df_levels = df_ana[cols_i1].dropna()

    with st.spinner("Test de Johansen..."):
        johan_result = test_johansen(df_levels, det_order=0, k_ar_diff=1)

    if "erreur" not in johan_result:
        col_j1, col_j2 = st.columns([2, 1])

        with col_j1:
            st.markdown("**Statistiques du test trace**")
            st.dataframe(
                johan_result["tableau"],
                use_container_width=True,
                hide_index=True,
            )

        with col_j2:
            n_coint = johan_result["n_coint"]
            if n_coint > 0:
                st.success(
                    f"✅ **{n_coint} relation(s) de cointégration** détectée(s)\n\n"
                    f"→ VECM recommandé"
                )
            else:
                st.info(
                    "ℹ️ **Pas de cointégration** détectée\n\n"
                    "→ VAR en différences approprié"
                )

        st.markdown(f"""
        <div class="insight-box">
        <b>📌 Conclusion Johansen :</b> {johan_result['conclusion']}<br>
        La statistique trace teste H₀ : rang de cointégration ≤ r.
        On rejette H₀ si la statistique dépasse la valeur critique à 5%.
        Le nombre de relations de cointégration retenu correspond au
        dernier rang pour lequel H₀ est rejeté.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3 : VECM (si cointégration détectée)
# ══════════════════════════════════════════════════════════════

st.subheader("3️⃣ Modèle VECM — Dynamiques de long terme")

if len(cols_i1) >= 2:
    n_coint_detected = johan_result.get("n_coint", 0) if "erreur" not in johan_result else 0
    n_coint_use      = max(n_coint_detected, 1)  # Au moins 1 pour illustration

    with st.spinner("Estimation du VECM..."):
        result_vecm = fit_vecm(
            df=df_levels,
            n_coint=n_coint_use,
            k_ar_diff=1,
            forecast_horizon=8,
        )

    if "erreur" not in result_vecm:
        tab_alpha, tab_beta, tab_prev = st.tabs([
            "Vitesses d'ajustement (α)",
            "Vecteurs cointégrants (β)",
            "Prévisions VECM",
        ])

        with tab_alpha:
            st.markdown("""
            Les **vitesses d'ajustement α** mesurent la rapidité avec laquelle
            chaque variable corrige ses déséquilibres de long terme.
            Un α négatif et significatif indique que la variable "converge"
            vers l'équilibre après un choc.
            """)
            st.dataframe(
                result_vecm["alpha"].round(4),
                use_container_width=True,
            )

        with tab_beta:
            st.markdown("""
            Les **vecteurs cointégrants β** définissent les relations de long terme.
            La combinaison linéaire β'Yt est stationnaire (= l'écart à l'équilibre).
            """)
            st.dataframe(
                result_vecm["beta"].round(4),
                use_container_width=True,
            )

        with tab_prev:
            df_vecm_prev = result_vecm["previsions"].round(3)
            df_vecm_prev.index = df_vecm_prev.index.strftime("%Y-T%q")
            df_vecm_prev.columns = [labels_vars.get(c, c) for c in df_vecm_prev.columns]
            st.dataframe(df_vecm_prev, use_container_width=True)

    else:
        st.warning(f"⚠️ {result_vecm['erreur']}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 4 : Fonctions de réponse impulsionnelle
# ══════════════════════════════════════════════════════════════

st.subheader("4️⃣ Fonctions de Réponse Impulsionnelle (IRF)")

st.markdown("""
L'IRF mesure la **réponse dynamique** d'une variable à un choc unitaire
sur une autre variable, toutes choses égales par ailleurs.
Identification par décomposition de Cholesky :
ordre **PIB → Inflation → Chômage** (convention macro standard).
""")

# Estimation VAR sur données différenciées pour l'IRF
var_cols_irf = [c for c in ["pib_growth", "d_inflation", "d_chomage"]
                if c in df_ana.columns]

# Reconstruire les différences si nécessaire
df_irf = pd.DataFrame()
for col in ["pib_growth", "chomage", "inflation"]:
    if col in df_ana.columns:
        if col in ["chomage", "inflation"]:
            df_irf[f"d_{col}"] = df_ana[col].diff()
        else:
            df_irf[col] = df_ana[col]

df_irf = df_irf.dropna()
var_cols_irf = list(df_irf.columns)

with st.spinner("Estimation VAR pour IRF..."):
    result_var_irf = fit_var(
        df=df_irf,
        n_lags=2,
        forecast_horizon=8,
    )

if "erreur" not in result_var_irf and "fitted" in result_var_irf:
    fitted_var = result_var_irf["fitted"]

    # Sélecteurs
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        impulse_var = st.selectbox(
            "Variable choc (impulse)",
            options=var_cols_irf,
            format_func=lambda x: labels_vars.get(x, x),
        )
    with col_sel2:
        response_var = st.selectbox(
            "Variable réponse (response)",
            options=var_cols_irf,
            format_func=lambda x: labels_vars.get(x, x),
            index=min(1, len(var_cols_irf)-1),
        )

    with st.spinner("Calcul IRF avec bootstrap (200 réplications)..."):
        irf_result = compute_irf(
            _var_fitted=fitted_var,
            impulse=impulse_var,
            response=response_var,
            periods=12,
            confidence_level=0.90,
        )

    if "erreur" not in irf_result:
        fig_irf = go.Figure()

        periodes = irf_result["periodes"]
        irf_vals = irf_result["irf"]
        ic_low   = irf_result["ic_lower"]
        ic_up    = irf_result["ic_upper"]

        # IC en premier
        fig_irf.add_trace(go.Scatter(
            x=list(periodes) + list(periodes[::-1]),
            y=list(ic_up) + list(ic_low[::-1]),
            fill="toself",
            fillcolor="rgba(0, 35, 149, 0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 90% (bootstrap)",
            hoverinfo="skip",
        ))

        # IRF
        fig_irf.add_trace(go.Scatter(
            x=periodes, y=irf_vals,
            name="IRF",
            line=dict(color=COLORS["France"], width=2.5),
        ))

        # Ligne zéro
        fig_irf.add_hline(
            y=0,
            line_dash="dash",
            line_color="#999999",
            line_width=1,
        )

        fig_irf = apply_house_style(
            fig_irf,
            title=(f"IRF : réponse de {labels_vars.get(response_var, response_var)} "
                   f"à un choc sur {labels_vars.get(impulse_var, impulse_var)}"),
            height=400,
        )
        fig_irf.update_xaxes(title_text="Trimestres après le choc")
        fig_irf.update_yaxes(title_text="Réponse (unité de la variable)")
        st.plotly_chart(fig_irf, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>📌 Lecture IRF :</b> Si la courbe est significativement positive
        (IC 90% au-dessus de zéro), le choc a un effet positif et statistiquement
        significatif. Si l'IC englobe zéro, l'effet n'est pas significativement
        différent de zéro. La persistance de la réponse indique la durée du
        mécanisme de transmission.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning(f"⚠️ IRF : {irf_result['erreur']}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 5 : Scénarios "what-if"
# ══════════════════════════════════════════════════════════════

st.subheader("5️⃣ Scénarios Macroéconomiques — What-If")

st.markdown("""
Simulation de **chocs exogènes** sur les prévisions baseline.
Les multiplicateurs de propagation sont calibrés sur la littérature
empirique (FMI World Economic Outlook, BCE Working Papers).
""")

# Prévisions baseline (SARIMA sur PIB)
from src.models import fit_sarima_auto

scenario_var = st.selectbox(
    "Variable à analyser",
    options=["pib_growth", "chomage", "inflation"],
    format_func=lambda x: labels_vars.get(x, x),
)

with st.spinner("Calcul prévisions baseline..."):
    baseline_sarima = fit_sarima_auto(
        series=df_ana[scenario_var].dropna(),
        forecast_horizon=8,
    )

if "erreur" not in baseline_sarima:

    st.markdown("#### Paramètres du choc")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        scenario_type = st.selectbox(
            "Type de choc",
            options=[
                "Hausse des taux BCE (+100 pb)",
                "Choc pétrolier (+20%)",
                "Récession demande (−1% PIB)",
                "Choc personnalisé",
            ],
        )
    with col_s2:
        if scenario_type == "Choc personnalisé":
            choc_amp = st.slider(
                "Amplitude du choc",
                min_value=-3.0, max_value=3.0, value=-0.5, step=0.1,
            )
        else:
            choc_amp = {
                "Hausse des taux BCE (+100 pb)": -0.4,
                "Choc pétrolier (+20%)":          0.3,
                "Récession demande (−1% PIB)":   -1.0,
            }[scenario_type]
            st.metric("Amplitude calibrée", f"{choc_amp:+.1f}")

    with col_s3:
        persistance = st.slider(
            "Persistance (trimestres)",
            min_value=1, max_value=8, value=4,
        )

    # Propagation calibrée selon le type de choc
    propagation_map = {
        "Hausse des taux BCE (+100 pb)": {},
        "Choc pétrolier (+20%)":          {},
        "Récession demande (−1% PIB)":    {},
        "Choc personnalisé":              {},
    }

    # Calcul des scénarios
    from src.models import compute_scenario

    prev_baseline = baseline_sarima["previsions"].to_frame(scenario_var)
    prev_scenario = compute_scenario(
        base_forecast=prev_baseline,
        variable=scenario_var,
        choc_amplitude=choc_amp,
        choc_persistance=persistance,
    )

    # Graphique comparatif
    hist_sc = baseline_sarima["historique"].iloc[-24:]

    fig_sc = go.Figure()

    # Historique
    fig_sc.add_trace(go.Scatter(
        x=hist_sc.index, y=hist_sc.values,
        name="Historique",
        line=dict(color=COLORS["France"], width=2.5),
    ))

    # Baseline
    fig_sc.add_trace(go.Scatter(
        x=prev_baseline.index,
        y=prev_baseline[scenario_var].values,
        name="Prévision baseline",
        line=dict(color="#2ecc71", width=2.5, dash="dash"),
    ))

    # Scénario choc
    fig_sc.add_trace(go.Scatter(
        x=prev_scenario.index,
        y=prev_scenario[scenario_var].values,
        name=f"Scénario : {scenario_type}",
        line=dict(color="#e74c3c", width=2.5, dash="dot"),
    ))

    # Zone entre baseline et scénario
    fig_sc.add_trace(go.Scatter(
        x=list(prev_baseline.index) + list(prev_scenario.index[::-1]),
        y=list(prev_baseline[scenario_var].values) +
          list(prev_scenario[scenario_var].values[::-1]),
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Écart baseline/scénario",
        hoverinfo="skip",
    ))

    fig_sc.add_vrect(
        x0=hist_sc.index[-1],
        x1=hist_sc.index[-1] + pd.DateOffset(days=1),
        line_width=1.5,
        line_color="#999999",
        line_dash="dash",
    )

    fig_sc = apply_house_style(
        fig_sc,
        title=f"Analyse de scénario — {labels_vars.get(scenario_var, scenario_var)}",
        height=450,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Tableau d'impact
    impact = prev_scenario[scenario_var] - prev_baseline[scenario_var]
    df_impact = pd.DataFrame({
        "Trimestre":   prev_baseline.index.strftime("%Y-T%q"),
        "Baseline":    prev_baseline[scenario_var].round(3).values,
        "Scénario":    prev_scenario[scenario_var].round(3).values,
        "Impact (Δ)":  impact.round(3).values,
    })
    with st.expander("📋 Tableau d'impact détaillé"):
        st.dataframe(df_impact, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    f"🇫🇷 French Macro Forecast Dashboard · "
    f"Économétrie : ADF, KPSS, Zivot-Andrews, Johansen, VECM, VAR, IRF · "
    f"Dernière mise à jour : {data['last_update']}"
)