# 🇫🇷 French Macro Forecast Dashboard

> **Tableau de bord macroéconomique interactif** — Analyse, modélisation
> et prévision de l'économie française dans son contexte européen.
> Données 1990–2026 · FRED · Banque Mondiale · Eurostat

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://french-macro-dashboard-lb5pdtru4eru3mxsjn7v54.streamlit.app)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-green)](https://www.statsmodels.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🔗 [👉 Accéder au dashboard en ligne](https://french-macro-dashboard-lb5pdtru4eru3mxsjn7v54.streamlit.app)

---

## 🎯 Présentation

Ce dashboard propose une **analyse macroéconomique professionnelle** de la France
et des principales économies de la zone euro (Allemagne, Italie, Espagne).
Il mobilise des méthodes économétriques avancées utilisées par les grandes institutions
(Banque de France, OCDE, FMI, BCE) pour produire des visualisations interactives,
des prévisions quantifiées et des analyses de scénarios.

Il illustre la maîtrise combinée de l'économétrie des séries temporelles,
du data engineering (APIs multiples) et de la visualisation de données.

---

## 📸 Aperçu

| Vue Générale | Prévisions SARIMA | Économétrie |
|---|---|---|
| KPIs temps réel, PIB, chômage | SARIMA auto + VAR + IC 90% | ADF, KPSS, Zivot-Andrews, IRF |

---

## 📊 Fonctionnalités

### 5 pages interactives

| Page | Contenu |
|------|---------|
| 📊 **Vue Générale** | KPIs temps réel, PIB, inflation, chômage, taux d'intérêt, spread OAT-Bund |
| 📈 **Conjoncture** | Output gap (filtre HP), cycle économique, taux de change EUR/USD |
| 🔮 **Prévisions** | SARIMA auto, VAR multivarié, intervalles de confiance 90%, scénarios |
| 🌍 **Comparaisons** | France vs Allemagne, Italie, Espagne, Zone Euro — 4 dimensions |
| 🔬 **Économétrie** | Tests ADF/KPSS/Zivot-Andrews, Johansen, VECM, IRF, scénarios what-if |

### Indicateurs couverts
- PIB réel (croissance trimestrielle et annuelle, base 100)
- Inflation IPCH harmonisée Eurostat (comparable entre pays)
- Taux de chômage harmonisé OCDE
- Taux souverains 10 ans (OAT, Bund) et spread OAT-Bund
- Taux directeur BCE, taux de change EUR/USD
- Output gap (filtre Hodrick-Prescott, λ=1600)

---

## 🔬 Méthodologie économétrique

### Tests de stationnarité
- **ADF** (Augmented Dickey-Fuller) avec sélection automatique des retards (BIC)
- **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) — hypothèses inversées
- **Zivot-Andrews** (1992) — ADF avec rupture structurelle endogène

> Les contradictions ADF/KPSS sont traitées par vote majoritaire (2/3 tests).
> Le test de Zivot-Andrews corrige le biais du test ADF en présence de ruptures
> structurelles (2008-2009, 2020).

### Modèles de prévision

| Modèle | Usage | Spécification |
|--------|-------|---------------|
| **SARIMA** | Prévision univariée | Sélection auto par grille AIC sur 18 combinaisons |
| **VAR(p)** | Prévision multivariée | Sélection retards : médiane AIC/BIC/HQIC, max 4 |
| **VECM** | Long terme cointégré | Johansen trace, det_order=0 |

### Fonctions de réponse impulsionnelle
Décomposition de Cholesky, ordre : PIB → Inflation → Chômage.
Intervalles de confiance par bootstrap (200 réplications).

### Output gap
Filtre Hodrick-Prescott (λ=1600, standard trimestriel).
*Note : approximation — la CE utilise une méthode par fonction de production.*

### Performance des modèles
- **RMSE SARIMA** (croissance PIB) : ~1,4–1,8 pp — comparable aux prévisions institutionnelles à court terme
- **Horizon de prévision** : 8 trimestres (2 ans)
- **Intervalles de confiance** : 90%, calculés analytiquement (SARIMA) et par simulation (VAR)

---

## 🗂️ Structure du projet

```
french-macro-dashboard/
├── app.py                    # Point d'entrée Streamlit
├── pages/
│   ├── 1_📊_Vue_Generale.py
│   ├── 2_📈_Conjoncture.py
│   ├── 3_🔮_Previsions.py
│   ├── 4_🌍_Comparaisons.py
│   └── 5_🔬_Econometrie.py
├── src/
│   ├── data_loader.py        # APIs FRED + World Bank
│   ├── data_processor.py     # Nettoyage, HP filter, output gap
│   ├── models.py             # SARIMA, VAR, VECM, IRF, tests
│   └── utils.py              # Graphiques Plotly réutilisables
├── config/
│   └── settings.py           # Codes séries, couleurs, paramètres
├── assets/
│   └── style.css             # Style Banque de France
├── packages.txt              # Dépendances système (Streamlit Cloud)
└── requirements.txt
```

---

## 🚀 Installation locale

### Prérequis
- Python 3.10+
- Clé API FRED gratuite sur [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/bduv/french-macro-dashboard.git
cd french-macro-dashboard

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et ajouter votre clé FRED_API_KEY

# 5. Lancer le dashboard
streamlit run app.py
```

---

## 📡 Sources de données

| Source | Indicateurs | Fréquence |
|--------|-------------|-----------|
| **FRED** (Fed St. Louis) | PIB, chômage, taux d'intérêt, taux de change | Mensuel / Trimestriel |
| **Eurostat via FRED** | Inflation IPCH harmonisée | Mensuel |
| **Banque Mondiale** | Dette publique, balance courante | Annuel |

---

## ⚠️ Avertissements

- Les prévisions sont produites à des fins **analytiques et pédagogiques**.
  Elles ne constituent pas un conseil en investissement ou en politique économique.
- Les résultats des tests économétriques dépendent de la période d'estimation
  et peuvent être sensibles aux points de rupture structurels.
- L'output gap calculé par filtre HP est une **approximation** ;
  la méthode officielle de la Commission Européenne utilise une
  fonction de production avec capital et travail.

---

## 👤 Auteur

**BdTchio** — Étudiant en économie quantitative  
[GitHub](https://github.com/bduv) · [Dashboard en ligne](https://french-macro-dashboard-lb5pdtru4eru3mxsjn7v54.streamlit.app)

---

## 📄 Licence

MIT License — voir [LICENSE](LICENSE)