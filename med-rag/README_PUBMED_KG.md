# PubMed → Knowledge Graph - Documentation

## Vue d'ensemble

Enrichissement automatique du Knowledge Graph avec des articles scientifiques de PubMed.

```
PubMed Search
    ↓
Articles (title, abstract, MeSH)
    ↓
NER (Entity Extraction)
    ↓
Knowledge Graph
    ↓
Supabase Persistence + Redis Cache
```

---

## Fonction principale

### `ingest_from_pubmed()`

Recherche PubMed et enrichit le KG en une seule étape.

**Signature** :
```python
from tools.kg_tool import ingest_from_pubmed

result = ingest_from_pubmed(
    query: str,                    # Requête PubMed
    max_results: int = 20,         # Nombre max d'articles
    mindate: str = "",             # Date min (YYYY ou YYYY/MM/DD)
    maxdate: str = "",             # Date max (YYYY ou YYYY/MM/DD)
    entity_types: list = None,     # Types d'entités à extraire
    provider: str = None,          # Provider NER (auto par défaut)
)
```

**Retour** :
```python
{
    "pubmed_query": "alzheimer treatment",
    "articles_found": 1250,           # Total dans PubMed
    "articles_processed": 20,         # Articles traités
    "entities_extracted": 145,        # Entités extraites
    "graph_stats": {
        "node_count": 523,
        "edge_count": 1847,
        "density": 0.0135
    }
}
```

---

## Exemples d'utilisation

### 1. Enrichissement basique

```python
from tools.kg_tool import ingest_from_pubmed, stats

# Avant
print("Before:", stats())

# Enrichir avec 50 articles sur Alzheimer
result = ingest_from_pubmed(
    query="alzheimer treatment",
    max_results=50
)

print(f"Processed: {result['articles_processed']} articles")
print(f"Extracted: {result['entities_extracted']} entities")
print("After:", stats())
```

### 2. Recherche avec filtres temporels

```python
# Articles récents (2024 uniquement)
result = ingest_from_pubmed(
    query="diabetes mellitus",
    max_results=30,
    mindate="2024",
    maxdate="2024"
)
```

### 3. Requêtes PubMed avancées

```python
# Utiliser les MeSH terms
result = ingest_from_pubmed(
    query="cardiovascular disease[MeSH] AND aspirin",
    max_results=40
)

# Filtrer par champ
result = ingest_from_pubmed(
    query="COVID-19[Title] AND vaccine[Abstract]",
    max_results=25
)

# Combiner plusieurs critères
result = ingest_from_pubmed(
    query="hypertension[MeSH] AND treatment[Title/Abstract]",
    mindate="2023",
    max_results=50
)
```

### 4. Enrichissement par domaine médical

```python
# Cardiologie
ingest_from_pubmed("myocardial infarction", max_results=50)
ingest_from_pubmed("heart failure", max_results=50)
ingest_from_pubmed("atrial fibrillation", max_results=50)

# Oncologie
ingest_from_pubmed("breast cancer", max_results=50)
ingest_from_pubmed("immunotherapy", max_results=50)

# Neurologie
ingest_from_pubmed("parkinson disease", max_results=50)
ingest_from_pubmed("multiple sclerosis", max_results=50)
```

---

## Workflow complet

### Étape 1 : Configuration

```bash
# .env
NCBI_API_KEY=your-ncbi-api-key        # Optionnel (augmente rate limit)
NCBI_EMAIL=your-email@example.com     # Recommandé par NCBI
NCBI_TOOL=med-assist                  # Nom de votre outil
```

### Étape 2 : Enrichissement initial

```python
from tools.kg_tool import ingest_from_pubmed

# Peupler le KG avec plusieurs domaines
domains = [
    ("cardiovascular disease", 50),
    ("diabetes mellitus", 50),
    ("hypertension", 40),
    ("cancer immunotherapy", 30),
    ("alzheimer disease", 40),
]

for query, max_results in domains:
    print(f"\nEnriching KG with: {query}")
    result = ingest_from_pubmed(query, max_results=max_results)
    print(f"  → {result['entities_extracted']} entities extracted")
```

### Étape 3 : Vérification

```python
from tools.kg_tool import stats, query_top_nodes, query_top_edges

# Stats globales
print(stats())

# Top entités
top_entities = query_top_nodes(n=20, sort_by="frequency")
for entity in top_entities:
    print(f"{entity['label']} ({entity['entity_type']}) - freq: {entity['frequency']}")

# Top relations
top_relations = query_top_edges(n=20)
for edge in top_relations:
    print(f"{edge['source']} → {edge['target']} (weight: {edge['weight']})")
```

### Étape 4 : Utilisation avec RAG

```python
from rag.chain import query_rag

# Le RAG bénéficie automatiquement du KG enrichi
answer = query_rag(
    question="What are the latest treatments for Alzheimer's disease?",
    conversation_id="medical_consultation",
    enable_kg_enrichment=True  # Active l'enrichissement KG
)

print(answer)
# → Réponse enrichie avec relations du KG PubMed
```

---

## Syntaxe de requête PubMed

### Champs de recherche

| Champ | Syntaxe | Exemple |
|-------|---------|---------|
| Titre | `[Title]` | `diabetes[Title]` |
| Abstract | `[Abstract]` | `treatment[Abstract]` |
| Titre/Abstract | `[Title/Abstract]` | `aspirin[Title/Abstract]` |
| MeSH terms | `[MeSH]` | `hypertension[MeSH]` |
| Auteur | `[Author]` | `Smith J[Author]` |
| Journal | `[Journal]` | `Nature[Journal]` |
| Date publication | `[pdat]` | `2024[pdat]` |

### Opérateurs booléens

```
AND  → diabetes AND treatment
OR   → diabetes OR hyperglycemia
NOT  → diabetes NOT type1
```

### Exemples de requêtes complexes

```python
# Articles récents sur un sujet précis
"COVID-19[MeSH] AND vaccine[Title] AND 2024[pdat]"

# Combiner plusieurs MeSH terms
"(diabetes mellitus[MeSH] OR hyperglycemia[MeSH]) AND treatment"

# Exclure certains types
"cancer[MeSH] NOT review[Publication Type]"

# Recherche par auteur et sujet
"Smith J[Author] AND alzheimer[MeSH]"
```

---

## Performance

### Temps de traitement

| Articles | Temps estimé | Entités extraites |
|----------|--------------|-------------------|
| 10 articles | ~10-15s | ~30-50 |
| 20 articles | ~20-30s | ~60-100 |
| 50 articles | ~45-60s | ~150-250 |
| 100 articles | ~90-120s | ~300-500 |

**Note** : Dépend de la vitesse du NER (GLiNER) et de la connexion NCBI.

### Rate limits NCBI

Sans API key : **3 requêtes/seconde**  
Avec API key : **10 requêtes/seconde**

**Recommandation** : Obtenir une clé API gratuite sur https://www.ncbi.nlm.nih.gov/account/

---

## Entités extraites

### Types d'entités médicales

Le NER extrait automatiquement :

- **DRUG** : Médicaments, molécules
- **DISEASE** : Maladies, pathologies
- **SYMPTOM** : Symptômes, signes cliniques
- **GENE** : Gènes
- **PROTEIN** : Protéines
- **ANATOMY** : Organes, structures anatomiques
- **PROCEDURE** : Procédures médicales

### Exemple d'extraction

**Article PubMed** :
> "Aspirin reduces the risk of myocardial infarction in patients with cardiovascular disease."

**Entités extraites** :
```python
{
    "DRUG": ["Aspirin"],
    "DISEASE": ["myocardial infarction", "cardiovascular disease"],
    "ANATOMY": ["patients"]
}
```

**Relations créées dans le KG** :
- `Aspirin` ↔ `myocardial infarction` (co-occurrence)
- `Aspirin` ↔ `cardiovascular disease` (co-occurrence)
- `myocardial infarction` ↔ `cardiovascular disease` (co-occurrence)

---

## Intégration avec RAG

### Comment le KG enrichit le RAG

**Sans KG** :
```
Query: "How does aspirin prevent heart attacks?"
→ Vector search only
→ Returns chunks with "aspirin" and "heart attack"
```

**Avec KG enrichi par PubMed** :
```
Query: "How does aspirin prevent heart attacks?"
→ Vector search + KG enrichment
→ Finds: aspirin, myocardial infarction, cardiovascular disease
→ KG shows relationships from PubMed articles
→ Re-ranks results with hybrid score
→ Adds context: "Related entities: Aspirin, Myocardial Infarction, Cardiovascular Disease"
```

### Bénéfices

✅ **Meilleure pertinence** : Documents avec entités liées sont favorisés  
✅ **Contexte enrichi** : Relations PubMed ajoutées au prompt LLM  
✅ **Découverte de liens** : Relations non explicites dans les documents uploadés  
✅ **Terminologie médicale** : MeSH terms standardisés

---

## Tests

### Test automatisé

```bash
python3 test_pubmed_kg.py
```

Ce script teste :
1. ✅ Import des modules
2. ✅ Stats KG avant enrichissement
3. ✅ Recherche PubMed
4. ✅ Ingestion dans le KG
5. ✅ Stats KG après enrichissement
6. ✅ Top entités extraites

### Test manuel

```python
from tools.kg_tool import ingest_from_pubmed, stats

# Avant
print("Before:", stats())

# Enrichir
result = ingest_from_pubmed("aspirin cardiovascular", max_results=10)

# Vérifier
print("After:", stats())
print(f"Entities: {result['entities_extracted']}")
```

---

## Dépannage

### Problème : "PubMed tool not available"

**Cause** : Import de `pubmed_tool.py` échoue  
**Solution** :
```bash
# Vérifier que le fichier existe
ls tools/pubmed_tool.py

# Vérifier les dépendances
pip install requests python-dotenv smolagents
```

### Problème : "Timeout while querying PubMed"

**Cause** : NCBI API lent ou indisponible  
**Solution** :
- Réduire `max_results`
- Réessayer plus tard
- Vérifier la connexion internet

### Problème : "HTTP 429 Too Many Requests"

**Cause** : Rate limit NCBI dépassé  
**Solution** :
- Ajouter `NCBI_API_KEY` dans `.env`
- Espacer les requêtes
- Réduire `max_results`

### Problème : "No articles retrieved from PubMed"

**Cause** : Requête trop spécifique ou aucun résultat  
**Solution** :
- Simplifier la requête
- Retirer les filtres de date
- Vérifier la syntaxe PubMed

### Problème : "Entities extracted: 0"

**Cause** : NER n'a pas trouvé d'entités médicales  
**Solution** :
- Vérifier que GLiNER est installé
- Tester avec une requête médicale claire
- Vérifier les logs NER

---

## Cas d'usage

### 1. Veille scientifique automatisée

```python
# Enrichir le KG avec les dernières publications
import schedule
import time

def weekly_pubmed_update():
    """Mise à jour hebdomadaire du KG."""
    topics = [
        "alzheimer disease",
        "diabetes mellitus",
        "cardiovascular disease"
    ]
    
    for topic in topics:
        ingest_from_pubmed(
            query=f"{topic}[MeSH]",
            mindate="2024/01/01",  # Articles de 2024
            max_results=20
        )

# Planifier tous les lundis
schedule.every().monday.at("09:00").do(weekly_pubmed_update)
```

### 2. Construction de KG spécialisé

```python
# KG spécialisé en cardiologie
cardio_topics = [
    "myocardial infarction",
    "heart failure",
    "atrial fibrillation",
    "coronary artery disease",
    "hypertension",
    "stroke"
]

for topic in cardio_topics:
    ingest_from_pubmed(f"{topic}[MeSH]", max_results=100)
```

### 3. Analyse de tendances

```python
# Comparer les publications par année
years = ["2020", "2021", "2022", "2023", "2024"]

for year in years:
    result = ingest_from_pubmed(
        query="COVID-19[MeSH]",
        mindate=year,
        maxdate=year,
        max_results=50
    )
    print(f"{year}: {result['articles_found']} articles found")
```

---

## Roadmap

- [ ] Support de PubMed Central (PMC) pour full-text
- [ ] Extraction de relations typées (TREATS, CAUSES, etc.)
- [ ] Filtrage par type de publication (RCT, Meta-analysis, etc.)
- [ ] Export des citations bibliographiques
- [ ] Détection de contradictions entre articles
- [ ] Scoring de qualité des articles (impact factor, citations)

---

## Ressources

- **NCBI E-utilities** : https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **PubMed Search** : https://pubmed.ncbi.nlm.nih.gov/
- **MeSH Database** : https://www.ncbi.nlm.nih.gov/mesh
- **API Key** : https://www.ncbi.nlm.nih.gov/account/
