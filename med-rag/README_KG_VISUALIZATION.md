# Knowledge Graph Visualization - Documentation

## Vue d'ensemble

Visualisation interactive du Knowledge Graph médical avec force-directed layout, filtres dynamiques et exploration des relations entre entités.

```
Backend (FastAPI)          Frontend (Next.js + React)
    ↓                              ↓
/kg/graph API         →    KnowledgeGraphViewer.tsx
    ↓                              ↓
NetworkX Graph        →    react-force-graph-2d
    ↓                              ↓
JSON (nodes + links)  →    Visualisation interactive
```

---

## Architecture

### Backend (FastAPI)

**Endpoints créés dans `main.py`** :

1. **GET /kg/stats** - Statistiques du KG
2. **GET /kg/graph** - Données pour visualisation (format node-link)
3. **GET /kg/node/{node_id}** - Détails d'un nœud
4. **GET /kg/top-nodes** - Top N nœuds

### Frontend (React)

**Composants créés** :

1. **`KnowledgeGraphViewer.tsx`** - Composant principal de visualisation
2. **`/knowledge-graph/page.tsx`** - Page Next.js

---

## Installation

### 1. Installer les dépendances

```bash
cd ui-med-rag
npm install react-force-graph-2d
```

### 2. Démarrer le backend

```bash
cd med-rag
python3 main.py
```

### 3. Démarrer le frontend

```bash
cd ui-med-rag
npm run dev
```

### 4. Accéder à la visualisation

Ouvrir : **http://localhost:3000/knowledge-graph**

---

## Utilisation

### Interface

**Header** :
- Titre : "Knowledge Graph"
- Stats : Nombre de nœuds et arêtes

**Filtres** :
- **Entity Type** : Filtrer par type (DRUG, DISEASE, SYMPTOM, etc.)
- **Max Nodes** : Limiter le nombre de nœuds affichés (10-500)
- **Min Frequency** : Filtrer les entités peu fréquentes (1-10)
- **Refresh** : Recharger les données

**Légende** :
- Couleurs par type d'entité :
  - 🔵 DRUG (bleu)
  - 🔴 DISEASE (rouge)
  - 🟠 SYMPTOM (orange)
  - 🟢 GENE (vert)
  - 🟣 PROTEIN (violet)
  - 🟡 ANATOMY (rose)
  - ⚪ UNKNOWN (gris)

**Graphe** :
- **Nœuds** : Taille proportionnelle à la fréquence
- **Liens** : Épaisseur proportionnelle au poids
- **Labels** : Nom de l'entité
- **Particules** : Animation sur les liens

**Interactions** :
- **Zoom** : Molette de la souris
- **Pan** : Cliquer-glisser
- **Click sur nœud** : Afficher détails + centrer
- **Hover** : Tooltip avec info

**Détails du nœud sélectionné** :
- Label
- Type (avec couleur)
- Fréquence
- Nombre de connexions

---

## API Endpoints

### GET /kg/stats

Retourne les statistiques du Knowledge Graph.

**Exemple** :
```bash
curl http://localhost:8000/kg/stats
```

**Réponse** :
```json
{
  "node_count": 9,
  "edge_count": 14,
  "connected_components": 2,
  "density": 0.3889
}
```

---

### GET /kg/graph

Retourne le graphe au format node-link pour visualisation.

**Paramètres** :
- `entity_type` (optionnel) : Filtrer par type (DRUG, DISEASE, etc.)
- `max_nodes` (défaut: 100) : Nombre max de nœuds
- `min_frequency` (défaut: 1) : Fréquence minimale

**Exemples** :
```bash
# Tous les nœuds (max 100)
curl http://localhost:8000/kg/graph

# Seulement les médicaments
curl http://localhost:8000/kg/graph?entity_type=DRUG

# Top 50 nœuds les plus fréquents
curl http://localhost:8000/kg/graph?max_nodes=50

# Entités avec fréquence >= 2
curl http://localhost:8000/kg/graph?min_frequency=2

# Combinaison de filtres
curl "http://localhost:8000/kg/graph?entity_type=DISEASE&max_nodes=30&min_frequency=2"
```

**Réponse** :
```json
{
  "nodes": [
    {
      "id": "aspirin",
      "label": "aspirin",
      "type": "DRUG",
      "frequency": 6,
      "degree": 5
    },
    {
      "id": "myocardial infarction",
      "label": "myocardial infarction",
      "type": "DISEASE",
      "frequency": 6,
      "degree": 5
    }
  ],
  "links": [
    {
      "source": "aspirin",
      "target": "myocardial infarction",
      "weight": 5,
      "relation_type": "co_occurrence"
    }
  ],
  "stats": {
    "total_nodes": 9,
    "total_edges": 14,
    "filtered": false
  }
}
```

---

### GET /kg/node/{node_id}

Retourne les détails d'un nœud et ses voisins.

**Exemple** :
```bash
curl http://localhost:8000/kg/node/aspirin
```

**Réponse** :
```json
{
  "node": {
    "id": "aspirin",
    "label": "aspirin",
    "entity_type": "DRUG",
    "frequency": 6
  },
  "neighbors": [
    {
      "id": "myocardial infarction",
      "label": "myocardial infarction",
      "entity_type": "DISEASE",
      "frequency": 6,
      "edge_weight": 5,
      "relation_type": "co_occurrence"
    }
  ]
}
```

---

### GET /kg/top-nodes

Retourne les top N nœuds par fréquence ou degré.

**Paramètres** :
- `n` (défaut: 20) : Nombre de nœuds
- `sort_by` (défaut: "frequency") : "frequency" ou "degree"

**Exemple** :
```bash
curl "http://localhost:8000/kg/top-nodes?n=10&sort_by=frequency"
```

**Réponse** :
```json
{
  "nodes": [
    {
      "id": "aspirin",
      "label": "aspirin",
      "entity_type": "DRUG",
      "frequency": 6,
      "degree": 5
    }
  ]
}
```

---

## Exemples d'utilisation

### 1. Visualiser tout le graphe

```
1. Ouvrir http://localhost:3000/knowledge-graph
2. Le graphe s'affiche automatiquement
3. Utiliser la souris pour zoomer/déplacer
```

### 2. Filtrer par type d'entité

```
1. Dans le menu "Entity Type", sélectionner "Drug"
2. Cliquer sur "Refresh"
3. Seuls les médicaments sont affichés
```

### 3. Explorer un nœud

```
1. Cliquer sur un nœud (ex: "aspirin")
2. Le graphe se centre sur le nœud
3. Un panneau de détails s'affiche à droite
4. Voir : type, fréquence, nombre de connexions
```

### 4. Réduire la complexité

```
1. Augmenter "Min Frequency" à 3
2. Réduire "Max Nodes" à 50
3. Cliquer "Refresh"
4. Graphe plus simple avec entités principales
```

---

## Cas d'usage

### Use Case 1 : Explorer les relations médicamenteuses

**Objectif** : Voir quelles maladies sont traitées par quels médicaments

**Étapes** :
1. Filtrer `entity_type=DRUG`
2. Cliquer sur un médicament (ex: "aspirin")
3. Observer les liens vers les maladies (DISEASE)
4. Identifier les co-occurrences fréquentes

**Résultat** : Comprendre le contexte d'utilisation d'un médicament

---

### Use Case 2 : Identifier les maladies centrales

**Objectif** : Trouver les maladies les plus mentionnées

**Étapes** :
1. Appeler `/kg/top-nodes?entity_type=DISEASE&n=10`
2. Ou filtrer `entity_type=DISEASE` dans l'UI
3. Observer la taille des nœuds (proportionnelle à la fréquence)

**Résultat** : Focus sur les maladies principales du corpus

---

### Use Case 3 : Découvrir des relations inattendues

**Objectif** : Trouver des liens non évidents entre entités

**Étapes** :
1. Afficher tout le graphe (sans filtres)
2. Observer les clusters (groupes de nœuds connectés)
3. Cliquer sur des nœuds périphériques
4. Identifier les chemins entre entités distantes

**Résultat** : Découverte de relations indirectes

---

### Use Case 4 : Comparer avant/après enrichissement PubMed

**Objectif** : Voir l'impact de l'ingestion PubMed

**Étapes** :
1. Noter les stats initiales (`/kg/stats`)
2. Enrichir avec PubMed : `ingest_from_pubmed("diabetes", max_results=50)`
3. Recharger la visualisation
4. Comparer le nombre de nœuds/arêtes

**Résultat** : Mesurer la croissance du KG

---

## Performance

### Limites recommandées

| Nombre de nœuds | Performance | Usage |
|-----------------|-------------|-------|
| < 50 | Excellente | Exploration détaillée |
| 50-100 | Bonne | Usage standard |
| 100-200 | Acceptable | Graphe complet |
| > 200 | Lente | Utiliser filtres |

### Optimisations

**1. Filtrer par fréquence**
```
min_frequency=2  → Retire les entités rares
min_frequency=3  → Garde seulement les entités fréquentes
```

**2. Limiter les nœuds**
```
max_nodes=50   → Top 50 entités
max_nodes=100  → Défaut (bon compromis)
```

**3. Filtrer par type**
```
entity_type=DRUG     → Seulement médicaments
entity_type=DISEASE  → Seulement maladies
```

---

## Troubleshooting

### Problème : Graphe vide

**Cause** : Knowledge Graph non peuplé

**Solution** :
```bash
cd med-rag
python3 -c "
from tools.kg_tool import ingest_from_pubmed
ingest_from_pubmed('aspirin cardiovascular', max_results=20)
"
```

---

### Problème : "Failed to fetch graph data"

**Cause** : Backend non démarré ou CORS

**Solution** :
1. Vérifier backend : `curl http://localhost:8000/kg/stats`
2. Vérifier CORS dans `main.py` (doit inclure `http://localhost:3000`)
3. Vérifier les logs backend

---

### Problème : Performance lente

**Cause** : Trop de nœuds

**Solution** :
- Réduire `max_nodes` à 50
- Augmenter `min_frequency` à 2 ou 3
- Filtrer par `entity_type`

---

### Problème : Nœuds qui se chevauchent

**Cause** : Force-directed layout pas encore stabilisé

**Solution** :
- Attendre quelques secondes (auto-stabilisation)
- Zoomer/dézoomer pour réinitialiser
- Recharger la page

---

## Roadmap

### Version actuelle (v1.0)
- ✅ Visualisation force-directed
- ✅ Filtres (type, fréquence, max nodes)
- ✅ Couleurs par type d'entité
- ✅ Détails au click
- ✅ Légende
- ✅ API endpoints

### Prochaines versions

**v1.1** :
- [ ] Recherche de nœud par nom
- [ ] Export (PNG, SVG, JSON)
- [ ] Historique de navigation

**v1.2** :
- [ ] Sous-graphe autour d'un nœud
- [ ] Shortest path entre 2 nœuds
- [ ] Clustering visuel (communautés)

**v1.3** :
- [ ] Timeline (évolution temporelle)
- [ ] Comparaison de graphes
- [ ] Annotations utilisateur

**v2.0** :
- [ ] Visualisation 3D (react-force-graph-3d)
- [ ] Graphe hiérarchique
- [ ] Intégration avec RAG (highlight entités de la réponse)

---

## Références

- **react-force-graph** : https://github.com/vasturiano/react-force-graph
- **NetworkX** : https://networkx.org/
- **D3.js force simulation** : https://d3js.org/d3-force
- **Next.js** : https://nextjs.org/

---

## Support

Pour toute question ou problème :
1. Vérifier cette documentation
2. Vérifier `INSTALL_KG_VISUALIZATION.md`
3. Vérifier les logs backend (`main.py`)
4. Vérifier la console navigateur (F12)
