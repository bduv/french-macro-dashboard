# ============================================================
# pages/2_📈_Conjoncture.py
# Analyse conjoncturelle détaillée — France
# Output gap, décomposition, taux de change
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader    import load_all_data
from src.data_processor import compute_output_gap, get_recession_bands, compute_hp_filter
from src.utils          import plot_area_chart, plot_multi_country_lines, apply_house_style
from config.settings    import COLORS

st.set_page_config(
    page_title="Conjoncture · French Macro Dashboard",
    page_icon="📈",
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

st.title("📈 Analyse Conjoncturelle — France")
st.markdown("*Cycle économique, output gap, finances publiques, taux de change*")
st.markdown("---")

with st.spinner("Chargement des données..."):
    data = load_all_data()

gdp   = data["gdp"]
rates = data["rates"]

# ══════════════════════════════════════════════════════════════
# SECTION 1 : Cycle économique & Output Gap
# ══════════════════════════════════════════════════════════════

st.subheader("🔄 Cycle économique français")

col1, col2 = st.columns(2)

with col1:
    # Décomposition HP : tendance vs cycle
    gdp_fr = gdp["France"].dropna()
    cycle, trend = compute_hp_filter(gdp_fr)

    fig_hp = go.Figure()
    fig_hp.add_trace(go.Scatter(
        x=gdp_fr.index, y=gdp_fr.values,
        name="PIB observé",
        line=dict(color=COLORS["France"], width=2),
    ))
    fig_hp.add_trace(go.Scatter(
        x=trend.index, y=trend.values,
        name="Tendance HP (PIB potentiel approx.)",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))
    fig_hp = apply_house_style(
        fig_hp,
        title="PIB réel vs Tendance Hodrick-Prescott",
        height=380
    )
    fig_hp.update_yaxes(title_text="Indice (base OCDE)")
    st.plotly_chart(fig_hp, use_container_width=True)

with col2:
    # Output gap
    output_gap = compute_output_gap(gdp_fr)
    fig_og = plot_area_chart(
        series=output_gap,
        title="Output Gap (filtre HP) — France",
        y_label="% du PIB potentiel",
        color="#002395",
        height=380,
    )
    st.plotly_chart(fig_og, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📌 Output Gap :</b> L'écart de production mesure la différence entre le PIB effectif
et le PIB potentiel (capacité productive maximale sans tensions inflationnistes).
Un output gap négatif persistant — comme observé après 2008 — indique une sous-utilisation
durable des capacités, justifiant une orientation accommodante de la politique économique.
<i>Note méthodologique : le filtre HP (λ=1600) constitue une approximation ; la Commission
Européenne utilise une méthode par fonction de production.</i>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2 : Taux de change & Compétitivité
# ══════════════════════════════════════════════════════════════

st.subheader("💱 Taux de change & Compétitivité")

eurusd = rates["EUR/USD"].dropna() if "EUR/USD" in rates.columns else None

if eurusd is not None and not eurusd.empty:
    fig_fx = plot_area_chart(
        series=eurusd,
        title="Taux de change EUR/USD (1 EUR = x USD)",
        y_label="EUR/USD",
        color="#003399",
        height=350,
    )
    # Ligne de parité
    fig_fx.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#e74c3c",
        line_width=1.5,
        annotation_text="Parité (1:1)",
        annotation_position="top left",
        annotation_font_color="#e74c3c",
    )
    st.plotly_chart(fig_fx, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>📌 EUR/USD :</b> Le taux de change de l'euro influence directement la compétitivité
    des exportateurs européens et le prix des importations énergétiques (libellées en dollars).
    La parité EUR/USD atteinte en 2022 — une première depuis 20 ans — a amplifié le choc
    inflationniste en renchérissant les importations de pétrole et de gaz.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3 : Tableau de bord de synthèse
# ══════════════════════════════════════════════════════════════

st.subheader("📋 Tableau de synthèse — Dernières valeurs")

# Construction d'un tableau récapitulatif
from src.utils import get_latest_value

indicateurs = {
    "Croissance PIB (g.a., %)":     gdp.get("France_growth_y", pd.Series()),
    "Taux de chômage (%)":          data["unemployment"].get("France", pd.Series()),
    "Inflation IPC (g.a., %)":      data["inflation"].get("France", pd.Series()),
    "OAT 10 ans (%)":               rates.get("OAT 10 ans (France)", pd.Series()),
    "Spread OAT-Bund (pb)":         rates.get("Spread OAT-Bund (pb)", pd.Series()),
    "EUR/USD":                      rates.get("EUR/USD", pd.Series()),
}

rows = []
for label, serie in indicateurs.items():
    if isinstance(serie, pd.Series) and not serie.empty:
        val, delta, date = get_latest_value(serie)
        rows.append({
            "Indicateur":          label,
            "Dernière valeur":     f"{val:.2f}" if val else "N/A",
            "Variation (1 an)":    f"{delta:+.2f}" if delta else "N/A",
            "Date":                str(date)[:10] if date else "N/A",
        })

if rows:
    import pandas as pd
    df_recap = pd.DataFrame(rows)
    st.dataframe(df_recap, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    f"🇫🇷 French Macro Forecast Dashboard · "
    f"Sources : FRED · Dernière mise à jour : {data['last_update']}"
)