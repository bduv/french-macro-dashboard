# ============================================================
# src/utils.py
# Fonctions utilitaires partagées entre toutes les pages
# Graphiques Plotly réutilisables + formatage
# ============================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import COLORS


# ══════════════════════════════════════════════════════════════
# SECTION 1 : Fonctions de base pour les graphiques
# ══════════════════════════════════════════════════════════════

def apply_house_style(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """
    Applique un style uniforme professionnel à tous les graphiques.
    Inspiré du style visuel de la Banque de France / BCE.
    """
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#1a1a2e", family="Inter, sans-serif"),
            x=0.01,
        ),
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#333333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e0e0e0",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            gridwidth=1,
            showline=True,
            linecolor="#cccccc",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            gridwidth=1,
            showline=False,
            zeroline=True,
            zerolinecolor="#cccccc",
            zerolinewidth=1,
        ),
        hovermode="x unified",
    )
    return fig


def add_recession_bands(fig: go.Figure, recession_bands: list) -> go.Figure:
    """
    Ajoute des zones grises sur le graphique pour marquer les récessions.
    Convention internationale : zones grises = récessions (NBER style).
    """
    for start, end in recession_bands:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="#cccccc",
            opacity=0.25,
            layer="below",
            line_width=0,
        )
    return fig


def add_zero_line(fig: go.Figure) -> go.Figure:
    """Ajoute une ligne horizontale en 0 (utile pour les taux de croissance)."""
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#999999",
        line_width=1,
        opacity=0.7,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SECTION 2 : Graphiques multi-pays
# ══════════════════════════════════════════════════════════════

def plot_multi_country_lines(
    df: pd.DataFrame,
    countries: list,
    title: str,
    y_label: str,
    recession_bands: list = None,
    add_zero: bool = False,
    height: int = 450,
) -> go.Figure:
    """
    Crée un graphique en lignes multi-pays avec le style maison.
    
    Arguments :
        df            : DataFrame avec les pays en colonnes
        countries     : Liste des pays à afficher
        title         : Titre du graphique
        y_label       : Label de l'axe Y
        recession_bands : Liste de tuples (début, fin) pour zones grises
        add_zero      : Ajouter une ligne en 0 ?
        height        : Hauteur du graphique
    """
    fig = go.Figure()

    for pays in countries:
        if pays not in df.columns:
            continue
        serie = df[pays].dropna()
        if serie.empty:
            continue

        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie.values,
            name=pays,
            line=dict(
                color=COLORS.get(pays, "#888888"),
                width=2.5,
            ),
            hovertemplate=f"<b>{pays}</b><br>%{{x|%Y-%m}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))

    if recession_bands:
        fig = add_recession_bands(fig, recession_bands)
    if add_zero:
        fig = add_zero_line(fig)

    fig = apply_house_style(fig, title=title, height=height)
    fig.update_yaxes(title_text=y_label)

    return fig


def plot_bar_comparison(
    df: pd.DataFrame,
    countries: list,
    title: str,
    y_label: str,
    last_n_years: int = 10,
    height: int = 400,
) -> go.Figure:
    """
    Crée un graphique en barres groupées pour comparer les pays
    sur les dernières années.
    """
    # Filtre sur les dernières années
    df_recent = df.last(f"{last_n_years * 12}ME") if len(df) > last_n_years * 4 else df

    fig = go.Figure()

    for pays in countries:
        if pays not in df_recent.columns:
            continue
        serie = df_recent[pays].dropna()
        if serie.empty:
            continue

        fig.add_trace(go.Bar(
            x=serie.index,
            y=serie.values,
            name=pays,
            marker_color=COLORS.get(pays, "#888888"),
            opacity=0.85,
            hovertemplate=f"<b>{pays}</b><br>%{{x|%Y-%m}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))

    fig = apply_house_style(fig, title=title, height=height)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=y_label)

    return fig


def plot_area_chart(
    series: pd.Series,
    title: str,
    y_label: str,
    color: str = "#002395",
    height: int = 350,
) -> go.Figure:
    """
    Graphique en aire remplie pour une seule série.
    Idéal pour le spread OAT-Bund ou l'output gap.
    """
    series_clean = series.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_clean.index,
        y=series_clean.values,
        fill="tozeroy",
        fillcolor=f"rgba{tuple(list(bytes.fromhex(color.lstrip('#'))) + [40])}",
        line=dict(color=color, width=2),
        name=y_label,
        hovertemplate=f"%{{x|%Y-%m}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
    ))

    fig = apply_house_style(fig, title=title, height=height)
    fig.update_yaxes(title_text=y_label)
    fig = add_zero_line(fig)

    return fig


def plot_heatmap_correlations(
    df: pd.DataFrame,
    title: str = "Matrice de corrélations",
    height: int = 450,
) -> go.Figure:
    """
    Heatmap des corrélations entre variables.
    Utile pour l'analyse économétrique exploratoire.
    """
    corr = df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{y} / %{x}<br>Corrélation : %{z:.2f}<extra></extra>",
    ))

    fig = apply_house_style(fig, title=title, height=height)
    fig.update_layout(margin=dict(l=100, r=30, t=60, b=100))

    return fig


# ══════════════════════════════════════════════════════════════
# SECTION 3 : KPIs (métriques clés en chiffres)
# ══════════════════════════════════════════════════════════════

def get_latest_value(series: pd.Series) -> tuple:
    """
    Retourne la dernière valeur disponible et sa variation.
    
    Retourne :
        (valeur, variation_sur_1_an, date)
    """
    series_clean = series.dropna()
    if series_clean.empty:
        return None, None, None

    latest_val  = series_clean.iloc[-1]
    latest_date = series_clean.index[-1]

    # Variation sur 1 an (ou 12 mois)
    if len(series_clean) >= 12:
        year_ago = series_clean.iloc[-12]
        delta    = latest_val - year_ago
    elif len(series_clean) >= 4:
        year_ago = series_clean.iloc[-4]
        delta    = latest_val - year_ago
    else:
        delta = None

    return round(latest_val, 2), round(delta, 2) if delta is not None else None, latest_date


def format_kpi_delta(delta: float, inverse: bool = False) -> str:
    """
    Formate la variation pour l'affichage KPI Streamlit.
    inverse=True : une baisse est positive (ex: chômage)
    """
    if delta is None:
        return "N/A"
    sign = "▲" if delta > 0 else "▼"
    return f"{sign} {abs(delta):.2f}"