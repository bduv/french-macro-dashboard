# ============================================================
# pages/4_🌍_Comparaisons.py
# Comparaisons internationales France vs Zone Euro
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_data
from src.utils       import (
    plot_multi_country_lines,
    plot_bar_comparison,
    apply_house_style,
    get_latest_value,
)
from config.settings import COLORS

st.set_page_config(
    page_title="Comparaisons · French Macro Dashboard",
    page_icon="🌍",
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

st.title("🌍 Comparaisons Européennes")
st.markdown("*France · Allemagne · Italie · Espagne · Zone Euro*")
st.markdown("---")

with st.spinner("Chargement..."):
    data  = load_all_data()

gdp   = data["gdp"]
unemp = data["unemployment"]
inf   = data["inflation"]
rates = data["rates"]

pays  = ["France", "Allemagne", "Italie", "Espagne", "Zone Euro"]

# ── KPIs comparatifs (dernières valeurs) ──────────────────────
st.subheader("📊 Snapshot — Dernières valeurs disponibles")

kpi_data = {}
for p in ["France", "Allemagne", "Italie", "Espagne"]:
    gdp_col = f"{p}_growth_y"
    kpi_data[p] = {
        "PIB (g.a.)":  get_latest_value(gdp[gdp_col])[0] if gdp_col in gdp.columns else None,
        "Chômage":     get_latest_value(unemp[p])[0] if p in unemp.columns else None,
        "Inflation":   get_latest_value(inf[p])[0] if p in inf.columns else None,
    }

cols_kpi = st.columns(4)
for i, (pays_k, vals) in enumerate(kpi_data.items()):
    with cols_kpi[i]:
        st.markdown(f"**{pays_k}**")
        for label, val in vals.items():
            if val is not None:
                st.metric(label, f"{val:.1f}%")

st.markdown("---")

# ── Graphiques de comparaison ─────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 PIB", "👥 Chômage", "💰 Inflation", "📉 Taux d'intérêt"
])

with tab1:
    pays_pib = [p for p in pays if p in gdp.columns]
    growth_cols = {p: f"{p}_growth_y" for p in pays_pib if f"{p}_growth_y" in gdp.columns}
    df_growth = gdp[[v for v in growth_cols.values()]].copy()
    df_growth.columns = list(growth_cols.keys())

    fig_gdp = plot_multi_country_lines(
        df=df_growth,
        countries=list(growth_cols.keys()),
        title="Croissance PIB réel — glissement annuel (%)",
        y_label="Variation annuelle (%)",
        add_zero=True,
        height=500,
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>📌 Divergences de croissance :</b> Depuis 2015, l'Espagne affiche
    la croissance la plus dynamique des grandes économies de la zone euro,
    portée par le tourisme et l'immobilier. L'Allemagne souffre depuis 2022
    d'une crise industrielle structurelle (énergie, automobile, exportations
    vers la Chine). La France résiste mieux grâce à sa diversification
    sectorielle et ses stabilisateurs automatiques.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    fig_unemp = plot_multi_country_lines(
        df=unemp,
        countries=[p for p in pays if p in unemp.columns],
        title="Taux de chômage harmonisé OCDE (%)",
        y_label="Taux de chômage (%)",
        height=500,
    )
    st.plotly_chart(fig_unemp, use_container_width=True)

    # Tableau des dernières valeurs
    latest_unemp = {}
    for p in ["France", "Allemagne", "Italie", "Espagne", "Zone Euro"]:
        if p in unemp.columns:
            val, delta, date = get_latest_value(unemp[p])
            latest_unemp[p] = {
                "Taux (%)": val,
                "Variation 1 an (pp)": delta,
                "Date": str(date)[:10] if date else "N/A",
            }
    st.dataframe(
        pd.DataFrame(latest_unemp).T.round(2),
        use_container_width=True,
    )

with tab3:
    fig_inf = plot_multi_country_lines(
        df=inf,
        countries=[p for p in pays if p in inf.columns],
        title="Inflation IPCH — glissement annuel (%)",
        y_label="Variation annuelle (%)",
        add_zero=True,
        height=500,
    )
    fig_inf.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#003399",
        line_width=1.5,
        annotation_text="Cible BCE : 2%",
        annotation_position="top right",
        annotation_font_color="#003399",
    )
    st.plotly_chart(fig_inf, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>📌 Convergence de l'inflation :</b> Après le pic inflationniste de
    2022 (France : 7,3%, Espagne : 10,8%, Allemagne : 11,6%), tous les pays
    ont convergé vers ou sous la cible BCE de 2% fin 2024. Cette désinflation
    rapide — la plus rapide depuis les années 1980 — s'explique par la
    normalisation des prix énergétiques et la transmission du resserrement
    monétaire. La France affiche structurellement l'inflation la plus faible
    grâce à la régulation des prix de l'énergie et du logement.
    </div>
    """, unsafe_allow_html=True)

with tab4:
    fig_rates = plot_multi_country_lines(
        df=rates[["OAT 10 ans (France)", "Bund 10 ans (Allemagne)"]].rename(columns={
            "OAT 10 ans (France)":      "OAT France",
            "Bund 10 ans (Allemagne)":  "Bund Allemagne",
        }),
        countries=["OAT France", "Bund Allemagne"],
        title="Taux souverains 10 ans (%)",
        y_label="Taux (%)",
        height=400,
    )
    fig_rates.data[0].line.color = COLORS["France"]
    fig_rates.data[1].line.color = COLORS["Allemagne"]
    st.plotly_chart(fig_rates, use_container_width=True)

    if "Spread OAT-Bund (pb)" in rates.columns:
        fig_spread = go.Figure()
        spread = rates["Spread OAT-Bund (pb)"].dropna()
        fig_spread.add_trace(go.Scatter(
            x=spread.index, y=spread.values,
            fill="tozeroy",
            fillcolor="rgba(192, 57, 43, 0.12)",
            line=dict(color="#c0392b", width=2),
            name="Spread OAT-Bund (pb)",
        ))
        fig_spread.add_hline(
            y=50, line_dash="dot",
            line_color="#e67e22", line_width=1.5,
            annotation_text="Seuil d'alerte (50 pb)",
        )
        fig_spread = apply_house_style(
            fig_spread,
            title="Spread OAT-Bund (points de base) — Prime de risque France",
            height=350,
        )
        st.plotly_chart(fig_spread, use_container_width=True)

st.markdown("---")
st.caption(
    f"🇫🇷 French Macro Forecast Dashboard · "
    f"Sources : FRED, Banque Mondiale · "
    f"Dernière mise à jour : {data['last_update']}"
)