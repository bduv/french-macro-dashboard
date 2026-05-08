# ============================================================
# app.py — Page d'accueil du dashboard
# ============================================================

import streamlit as st
import os

st.set_page_config(
    page_title="French Macro Dashboard",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🇫🇷 French Macro Forecast Dashboard")
st.markdown("### *Analyse macroéconomique de la France et de la Zone Euro*")
st.markdown("---")

st.markdown("""
Ce dashboard propose une analyse macroéconomique complète et interactive de l'économie française,
replacée dans son contexte européen. Il mobilise des données officielles issues de la
**Federal Reserve (FRED)** et de la **Banque Mondiale**, couvrant la période 1990–2026.
""")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.info("**📊 Vue Générale**\nKPIs, PIB, chômage, inflation, taux")
with col2:
    st.info("**📈 Conjoncture**\nCycle, output gap, taux de change")
with col3:
    st.info("**🔮 Prévisions**\nARIMA, VAR, intervalles de confiance")
with col4:
    st.info("**🌍 Comparaisons**\nFrance vs Zone Euro")
with col5:
    st.info("**🔬 Économétrie**\nTests ADF, cointégration, VAR")

st.markdown("---")
st.markdown("""
**Sources :** Federal Reserve Bank of St. Louis (FRED) · Banque Mondiale  
**Méthodologie :** Les prévisions reposent sur des modèles VAR (Vector AutoRegression)
estimés sur données trimestrielles. Les intervalles de confiance sont calculés
par simulation Monte Carlo (500 tirages).  
**Avertissement :** Ce dashboard est produit à des fins analytiques et pédagogiques.
Il ne constitue pas un conseil en investissement.
""")

st.caption("French Macro Forecast Dashboard · v1.0 · Mai 2026")