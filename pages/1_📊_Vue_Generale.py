# ============================================================
# pages/1_📊_Vue_Generale.py
# Tableau de bord principal : KPIs + graphiques macro France
# ============================================================

import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader   import load_all_data
from src.data_processor import get_recession_bands, compute_output_gap
from src.utils          import (
    plot_multi_country_lines,
    plot_area_chart,
    get_latest_value,
)
from config.settings import COLORS

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Vue Générale · French Macro Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Chargement du CSS ─────────────────────────────────────────
def load_css():
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "style.css"
    )
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── En-tête ───────────────────────────────────────────────────
st.title("📊 Vue Générale — France & Zone Euro")
st.markdown(
    "*Indicateurs macroéconomiques clés · Données : FRED, Banque Mondiale · "
    "Mise à jour automatique*"
)
st.markdown("---")

# ── Chargement des données ────────────────────────────────────
with st.spinner("Chargement des données..."):
    data = load_all_data()

gdp    = data["gdp"]
unemp  = data["unemployment"]
inf    = data["inflation"]
rates  = data["rates"]

# ══════════════════════════════════════════════════════════════
# SECTION 1 : KPIs France (dernières valeurs disponibles)
# ══════════════════════════════════════════════════════════════

st.subheader("🇫🇷 Indicateurs clés — France (dernière observation)")

col1, col2, col3, col4, col5 = st.columns(5)

# PIB — croissance annuelle
# ── KPI PIB : glissement annuel du dernier trimestre disponible ──
# Note : peut légèrement différer du chiffre annuel INSEE (+1.2% pour 2024)
# car on mesure T4-2024/T4-2023 et non la moyenne annuelle 2024/2023
gdp_growth_fr = gdp.get("France_growth_y", pd.Series(dtype=float))

val_gdp, delta_gdp, date_gdp = get_latest_value(gdp_growth_fr)

# Affichage avec note méthodologique
with col1:
    st.metric(
        label="Croissance PIB (g.a. trimestrielle)",
        value=f"{val_gdp:.1f} %" if val_gdp else "N/A",
        delta=f"{delta_gdp:.1f} pp" if delta_gdp else None,
        help="Variation du PIB réel T vs T-4 trimestres. "
             "Différent de la croissance annuelle moyenne (+1,2% en 2024 selon INSEE)."
    )

# Chômage
val_un, delta_un, date_un = get_latest_value(unemp["France"])

# Inflation
val_inf, delta_inf, date_inf = get_latest_value(inf["France"])

# OAT 10 ans
val_oat, delta_oat, date_oat = get_latest_value(rates["OAT 10 ans (France)"])

# Spread OAT-Bund
if "Spread OAT-Bund (pb)" in rates.columns:
    val_spread, delta_spread, date_spread = get_latest_value(rates["Spread OAT-Bund (pb)"])
else:
    val_spread, delta_spread, date_spread = None, None, None

with col2:
    st.metric(
        label="Taux de chômage",
        value=f"{val_un:.1f} %" if val_un else "N/A",
        delta=f"{delta_un:.1f} pp" if delta_un else None,
        delta_color="inverse",  # baisse du chômage = positif
    )

with col3:
    st.metric(
        label="Inflation (IPC g.a.)",
        value=f"{val_inf:.1f} %" if val_inf else "N/A",
        delta=f"{delta_inf:.1f} pp" if delta_inf else None,
        delta_color="off",  # neutre — désinflation = ambivalent
    )

with col4:
    st.metric(
        label="OAT 10 ans",
        value=f"{val_oat:.2f} %" if val_oat else "N/A",
        delta=f"{delta_oat:.2f} pp" if delta_oat else None,
        delta_color="inverse",
    )

with col5:
    st.metric(
        label="Spread OAT-Bund",
        value=f"{val_spread:.0f} pb" if val_spread else "N/A",
        delta=f"{delta_spread:.0f} pb" if delta_spread else None,
        delta_color="inverse",
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2 : Croissance du PIB
# ══════════════════════════════════════════════════════════════

st.subheader("📈 Croissance du PIB réel")

# Récessions France pour zones grises
recession_bands = []
if "France_growth_q" in gdp.columns:
    recession_bands = get_recession_bands(gdp["France_growth_q"].dropna())

tab_gdp1, tab_gdp2 = st.tabs(["Glissement annuel (%)", "Indice en base 100 (1995)"])

with tab_gdp1:
    # Construction explicite du DataFrame de croissance annuelle
    pays_dispo = []
    df_growth  = pd.DataFrame()

    for pays in ["France", "Allemagne", "Italie", "Espagne", "Zone Euro"]:
        col = f"{pays}_growth_y"
        if col in gdp.columns:
            serie = gdp[col].dropna()
            if not serie.empty:
                df_growth[pays] = gdp[col]
                pays_dispo.append(pays)

    if not df_growth.empty:
        fig_gdp = plot_multi_country_lines(
            df=df_growth,
            countries=pays_dispo,
            title="Taux de croissance du PIB réel — glissement annuel (%)",
            y_label="Variation annuelle (%)",
            recession_bands=recession_bands,
            add_zero=True,
            height=480,
        )
        st.plotly_chart(fig_gdp, use_container_width=True)
    else:
        st.warning("Données de croissance PIB indisponibles.")

with tab_gdp2:
    from src.data_processor import normalize_index
    pays_pib   = ["France", "Allemagne", "Italie", "Espagne", "Zone Euro"]
    cols_dispo = [p for p in pays_pib if p in gdp.columns]
    df_pib_norm = normalize_index(gdp[cols_dispo].copy(), base_date="1995-01-01")

    fig_norm = plot_multi_country_lines(
        df=df_pib_norm,
        countries=cols_dispo,
        title="PIB réel en base 100 (1995) — comparaison niveaux",
        y_label="Indice (base 100 = 1995)",
        recession_bands=recession_bands,
        height=480,
    )
    st.plotly_chart(fig_norm, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📌 Lecture :</b> Le graphique en base 100 permet de comparer les <i>niveaux</i> de PIB
indépendamment des tailles d'économie. L'Espagne et la France affichent une trajectoire
similaire depuis 1995, tandis que l'Allemagne a creusé un écart structurel à partir de
2005 grâce à ses réformes Hartz. Le décrochage collectif de 2020 (Covid) est suivi d'un
rebond asymétrique : l'Espagne, plus dépendante du tourisme, a récupéré plus lentement.
Les zones grises indiquent les récessions techniques (deux trimestres consécutifs négatifs).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3 : Chômage
# ══════════════════════════════════════════════════════════════

st.subheader("👥 Taux de chômage harmonisé (OCDE)")

fig_unemp = plot_multi_country_lines(
    df=unemp,
    countries=["France", "Allemagne", "Italie", "Espagne", "Zone Euro"],
    title="Taux de chômage harmonisé — comparaison européenne (%)",
    y_label="Taux de chômage (%)",
    recession_bands=recession_bands,
    height=450,
)
st.plotly_chart(fig_unemp, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📌 Lecture :</b> L'Espagne affiche structurellement le chômage le plus élevé de la zone
euro. La France se stabilise à <b>7,3% au T4 2024</b>, son niveau le plus bas depuis 1982
(hors T4 2022). Toutefois, cette amélioration masque une hausse du chômage des jeunes
(15-24 ans remontant à 21,5% fin 2024) et une progression du halo autour du chômage,
signaux d'une dégradation conjoncturelle naissante en fin d'année.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 4 : Inflation
# ══════════════════════════════════════════════════════════════

st.subheader("💰 Inflation — Indice des Prix à la Consommation")

fig_inf = plot_multi_country_lines(
    df=inf,
    countries=["France", "Allemagne", "Italie", "Espagne"],
    title="Inflation IPC — glissement annuel (%)",
    y_label="Variation annuelle (%)",
    recession_bands=recession_bands,
    add_zero=True,
    height=450,
)

# Ligne cible BCE à 2%
import plotly.graph_objects as go
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
<b>📌 Lecture :</b> La cible d'inflation de la BCE est fixée à 2% à moyen terme (symétrique
depuis juillet 2021). Le choc inflationniste de 2021-2022 a contraint la BCE à relever
ses taux directeurs de <b>450 points de base entre juillet 2022 et septembre 2023</b> —
cycle de resserrement le plus rapide de son histoire. Depuis juin 2024, la BCE est entrée
dans un cycle d'assouplissement progressif (−25 pb par réunion). L'inflation en France
est revenue à <b>1,8% en décembre 2024</b> (IPCH), en dessous de la cible BCE,
portée par la désinflation des services et la baisse des prix énergétiques.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 5 : Taux d'intérêt & Spread
# ══════════════════════════════════════════════════════════════

st.subheader("📉 Taux d'intérêt & Risque souverain")

col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_rates = plot_multi_country_lines(
        df=rates[["OAT 10 ans (France)", "Bund 10 ans (Allemagne)", "BCE (taux directeur)"]].rename(
            columns={
                "OAT 10 ans (France)":      "OAT 10 ans",
                "Bund 10 ans (Allemagne)":  "Bund 10 ans",
                "BCE (taux directeur)":     "BCE",
            }
        ),
        countries=["OAT 10 ans", "Bund 10 ans", "BCE"],
        title="Taux d'intérêt (%)",
        y_label="Taux (%)",
        height=380,
    )
    # Override couleurs manuellement pour ce graphique
    fig_rates.data[0].line.color = COLORS["France"]
    fig_rates.data[1].line.color = COLORS["Allemagne"]
    if len(fig_rates.data) > 2:
        fig_rates.data[2].line.color = COLORS["Zone Euro"]
    st.plotly_chart(fig_rates, use_container_width=True)

with col_r2:
    if "Spread OAT-Bund (pb)" in rates.columns:
        fig_spread = plot_area_chart(
            series=rates["Spread OAT-Bund (pb)"],
            title="Spread OAT 10 ans − Bund 10 ans (points de base)",
            y_label="Points de base",
            color="#c0392b",
            height=380,
        )
        st.plotly_chart(fig_spread, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📌 Spread OAT-Bund :</b> Cet écart mesure la prime de risque souverain de la France
face à l'Allemagne. Structurellement autour de 30-50 pb avant 2022, il a franchi
<b>80 pb à l'été 2024</b> dans le contexte de la dissolution de l'Assemblée nationale
et des incertitudes budgétaires. Cette persistance au-delà de 80 pb reflète l'inquiétude
des marchés face à un déficit public atteignant <b>6,1% du PIB en 2024</b>,
loin de l'objectif des 3% du Pacte de stabilité.
</div>
""", unsafe_allow_html=True)

# ── Pied de page ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"🇫🇷 French Macro Forecast Dashboard · "
    f"Sources : FRED (Federal Reserve), Banque Mondiale · "
    f"Dernière mise à jour : {data['last_update']}"
)