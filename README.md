<p align="center">
  <img src="outputs/images/logo the north face.png" alt="Logo The North Face" width="320">
</p>

## **The North Face — Machine learning non supervisé**


### **Description du projet**

The North Face est une marque américaine d'équipement outdoor, fondée en 1968 pour équiper les grimpeurs. Elle propose vêtements, chaussures et matériel de plein air. Depuis les années 1990, sa clientèle s'est élargie au-delà des pratiquants, et la marque est devenue un symbole de style dans les années 2000.

Le service marketing souhaite exploiter le machine learning pour renforcer les ventes en ligne sur [thenorthface.fr](https://www.thenorthface.fr/). Deux axes ont été identifiés :

1. **Système de recommandation**: proposer des produits similaires à ceux consultés par l'utilisateur, par exemple via une section du type "Vous pourriez aussi aimer…" sur chaque fiche produit.
2. **Structuration du catalogue**: extraire des thématiques latentes dans les descriptions pour questionner les catégories existantes et envisager une navigation plus pertinente.

Ce dépôt couvre un pipeline NLP complet sur un catalogue de descriptions produits : nettoyage du texte, vectorisation TF-IDF, clustering, recommandation hybride et topic modeling par LSA.

### **Objectifs**

Le projet se décline en trois volets :

1. Identifier des **groupes de produits** aux descriptions proches.
2. S'appuyer sur ces regroupements pour construire un **algorithme de recommandation** simple.
3. Appliquer le **topic modeling** pour estimer les thèmes latents présents dans les descriptions.

### **Périmètre**

Travail sur un corpus de descriptions issues du catalogue The North Face. Les données brutes sont fournies dans le dépôt ; le périmètre couvre l'exploration, le prétraitement, la modélisation et la restitution des résultats dans des notebooks Jupyter et des artefacts exportés.

---

## Jeu de données

| Élément | Détail |
| --- | --- |
| Fichier source | `data/sample-data.csv` |
| Volume | 500 produits |
| Colonnes | `id`, `description` |
| Langue | Anglais |
| Nature du texte | Descriptions marketing avec balises HTML (`<br>`, listes, gras, etc.), mentions de matériaux, poids, pays de fabrication, programme de recyclage |

Après prétraitement, les données nettoyées sont exportées dans `outputs/data/cleaned_data.csv` avec la colonne `clean_description`. La matrice TF-IDF et le vectorizer sont sauvegardés dans `outputs/models/tfidf_artifacts.pkl`.

---

## Structure du dépôt

```text
projet-the-north-face-bloc3/
├── data/
│   └── sample-data.csv                    # Catalogue brut (id, description)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb         # EDA, nettoyage spaCy, TF-IDF, exports
│   └── 02_models_training.ipynb         # Clustering, recommandation, LSA
├── outputs/
│   ├── data/
│   │   └── cleaned_data.csv               # Textes nettoyés exportés
│   ├── images/                            # Graphiques et visualisations
│   │   ├── logo the north face.png
│   │   ├── 01_len_descriptions_dist.png
│   │   ├── 02_raw_text_wc.png
│   │   ├── 03_word_count_comparison.png
│   │   ├── 04_kmeans_elbow_silhouette_calinski.png
│   │   ├── 05_kmeans_vs_dbscan_dist.png
│   │   ├── 06_lsa_topic_dist.png
│   │   ├── 07_lsa_wordclouds.png
│   │   └── 08_dbscan_lsa_cross.png
│   └── models/
│       └── tfidf_artifacts.pkl            # Matrice TF-IDF, vectorizer, métadonnées
├── src/
│   └── .gitkeep                           # Espace réservé au code applicatif
├── main.py                                # Point d'entrée Python minimal
├── pyproject.toml                         # Dépendances et métadonnées du projet
├── requirements.txt                       # Dépendances figées pour pip
├── uv.lock                                # Verrouillage des versions (uv)
├── .python-version                        # Version Python cible (3.12)
├── LICENSE
└── README.md
```

### Rôle des notebooks

| Notebook | Contenu principal |
| --- | --- |
| `01_eda_preprocessing.ipynb` | Statistiques descriptives, WordClouds, nettoyage HTML, lemmatisation spaCy, stop words métier, vectorisation TF-IDF, export CSV et artefacts |
| `02_models_training.ipynb` | KMeans (baseline), DBSCAN (cosinus), comparaison des modèles, fonction `find_similar_items`, LSA (`TruncatedSVD`), analyses croisées clusters / topics |

### Visualisations générées

Les figures produites par les notebooks sont enregistrées dans `outputs/images/`, notamment :

- distributions de longueur des descriptions ;
- WordClouds avant / après preprocessing ;
- courbes coude, silhouette et Calinski-Harabasz pour KMeans ;
- comparaison des distributions KMeans vs DBSCAN ;
- variance expliquée LSA, WordClouds par topic ;
- heatmap de correspondance clusters DBSCAN / topics LSA.

---

## **Configuration et installation**

### **Prérequis**

- Python **3.12**
- Un gestionnaire d'environnement : [uv](https://docs.astral.sh/uv/) (recommandé) ou `pip` avec `venv`

### **Installation**

#### **Option 1 - avec uv**

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

#### **Option 2 - avec pip et requirements.txt**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Sous Windows (PowerShell) : `.\.venv\Scripts\Activate.ps1` à la place de `source .venv/bin/activate`.

Le fichier `requirements.txt` reprend les dépendances verrouillées du projet pour une installation reproductible avec `pip`.

### **Dépendances principales**

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (TF-IDF, KMeans, DBSCAN, TruncatedSVD, métriques)
- `spacy` et modèle `en_core_web_sm`
- `wordcloud`

### **Exécution**

1. Lancer `notebooks/01_eda_preprocessing.ipynb` pour produire `cleaned_data.csv` et `tfidf_artifacts.pkl`.
2. Lancer `notebooks/02_models_training.ipynb` pour le clustering, la recommandation et le topic modeling.

Les chemins relatifs des notebooks supposent une exécution depuis le dossier `notebooks/`.

---

## **Méthodologie**

### **Prétraitement textuel**

- Suppression du HTML et normalisation du texte.
- Tokenisation, lemmatisation et filtrage des stop words avec **spaCy**.
- Enrichissement des stop words avec des termes récurrents du domaine (matériaux, recyclage, pays, etc.).
- Encodage des descriptions via **TF-IDF** (`TfidfVectorizer`, unigrammes et bigrammes, plafond de features).

### **Partie 1. Clustering**

- **KMeans** comme baseline, avec choix de `k` via inertie, silhouette et score de Calinski-Harabasz.
- **DBSCAN** sur la matrice TF-IDF avec métrique **cosinus**, recherche d'hyperparamètres visant environ **10 à 20 clusters** et un taux d'outliers raisonnable.
- Analyse qualitative par cluster (mots dominants, WordClouds).

### **Partie 2. Recommandation**

- Fonction **`find_similar_items(item_id)`** : recommandations hybrides combinant appartenance au cluster DBSCAN et **similarité cosinus** sur les vecteurs TF-IDF.
- Gestion des outliers (cluster `-1`) par similarité globale sur le catalogue.
- Démonstration dans le notebook sur plusieurs identifiants produits.

### **Partie 3. Topic modeling**

- **LSA** via `TruncatedSVD` sur la matrice TF-IDF, avec un nombre de composantes dans une plage interprétable (environ 10 à 20 topics).
- Matrice encodée stockée dans **`topic_encoded_df`**.
- Attribution d'un **topic principal** par document pour faciliter l'interprétation.
- WordClouds par topic et comparaison avec les clusters.

---

## **Livrables**

Conformément au cahier des charges du projet :

| Livrable | Réalisation dans ce dépôt |
| --- | --- |
| Au moins un modèle de clustering et des WordClouds par cluster | KMeans et DBSCAN dans `02_models_training.ipynb` ; visualisations dans `outputs/images/` |
| Code permettant de saisir un `id` produit et d'obtenir des articles similaires | Fonction `find_similar_items` dans `02_models_training.ipynb` (démonstration sur des `id` de test) |
| Au moins un modèle `TruncatedSVD` et des WordClouds des topics latents | LSA dans `02_models_training.ipynb` ; exports graphiques associés |
| Données et artefacts intermédiaires | `outputs/data/cleaned_data.csv`, `outputs/models/tfidf_artifacts.pkl` |

---

## **Contexte et auteur**

- Contexte : projet réalisé dans le cadre de la certification **CDSD — Bloc 3 (Unsupervised Machine Learning)** à **Jedha**, sur un cas d'usage e-commerce inspiré de The North Face.

- Auteur : **RANJAKASOA Raphaël Marcellin**

---

## Licence

Voir le fichier `LICENSE` à la racine du dépôt.
