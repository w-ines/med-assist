# Knowledge Graph - Documentation Complète

## Architecture

Le système de Knowledge Graph (KG) utilise une architecture à 3 niveaux pour optimiser les performances :

```
┌─────────────────────────────────────────────────┐
│          tools/kg_tool.py (API)                 │
│  - ingest_text()   : Extraction NER + KG        │
│  - query_graph()   : Recherche dans le KG       │
│  - stats()         : Statistiques du graphe     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           kg/store.py (Logic)                   │
│  - persist_graph() : Supabase + invalidate cache│
│  - load_graph()    : Redis → Supabase fallback  │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
┌────────▼────────┐  ┌───────▼──────────┐
│  Redis Cache    │  │  Supabase DB     │
│  TTL: 1h        │  │  kg_nodes        │
│  Speedup: 20x   │  │  kg_edges        │
└─────────────────┘  └──────────────────┘
```

## Composants

### 1. Persistence (Supabase)

**Tables** :
- `kg_nodes` : Entités médicales (DRUG, DISEASE, etc.)
- `kg_edges` : Relations entre entités (co-occurrence)

**Fichiers** :
- `storage/kg_repository.py` : CRUD pour Supabase
- `storage/supabase_client.py` : Client centralisé
- `migrations/kg_tables.sql` : Migration SQL

**Configuration** (.env) :
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### 2. Cache (Redis)

**Fonctionnalités** :
- Cache du graphe complet (TTL 1h)
- Invalidation automatique après écriture
- Speedup mesuré : **20x plus rapide**

**Fichiers** :
- `storage/kg_cache_redis.py` : Gestion du cache Redis

**Configuration** (.env) :
```bash
REDIS_URL=redis://localhost:6379/0
```

**Installation Redis** :
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

### 3. Extraction NER

**Providers supportés** :
- `gliner` : GLiNER2 (par défaut, meilleur pour biomédical)
- `spacy` : spaCy NER (fallback)

**Types d'entités** :
- `DRUG` : Médicaments
- `DISEASE` : Maladies
- `SYMPTOM` : Symptômes
- `GENE` : Gènes
- `PROTEIN` : Protéines

## Utilisation

### Ingest de texte

```python
from tools.kg_tool import ingest_text

result = ingest_text(
    text="Aspirin reduces myocardial infarction risk",
    source="PMID_12345",
    provider="gliner",  # ou "spacy"
    entity_types=["DRUG", "DISEASE"]
)

print(result)
# {
#   'ner': {
#     'entities': {
#       'DRUG': [{'text': 'Aspirin', 'confidence': 0.95}],
#       'DISEASE': [{'text': 'myocardial infarction', 'confidence': 0.92}]
#     },
#     'provider': 'gliner'
#   },
#   'graph_stats': {
#     'node_count': 2,
#     'edge_count': 1,
#     'connected_components': 1,
#     'density': 1.0
#   }
# }
```

### Statistiques du graphe

```python
from tools.kg_tool import stats

print(stats())
# {
#   'node_count': 150,
#   'edge_count': 320,
#   'connected_components': 5,
#   'density': 0.028
# }
```

### Requête sur le graphe

```python
from tools.kg_tool import query_graph

# Rechercher tous les médicaments
drugs = query_graph(entity_type="DRUG", limit=10)

# Rechercher par label
aspirin = query_graph(label="Aspirin")

# Rechercher les voisins d'une entité
neighbors = query_graph(node_id="aspirin", neighbors=True)
```

## Performance

### Benchmarks

| Opération | Sans cache | Avec cache | Speedup |
|-----------|-----------|------------|---------|
| load_graph() | 0.75ms | 0.04ms | **20x** |
| stats() | 0.80ms | 0.05ms | **16x** |
| query_graph() | 1.20ms | 0.08ms | **15x** |

### Optimisations

1. **Cache Redis** : Graphe complet en mémoire (TTL 1h)
2. **Index Supabase** : Index sur `entity_type`, `frequency`, `weight`
3. **Batch upsert** : Insertion par lots pour réduire les appels réseau

## Workflow de persistence

### Écriture (ingest)

```
1. Extraction NER du texte
2. Construction du graphe NetworkX
3. Upsert nodes → Supabase (kg_nodes)
4. Upsert edges → Supabase (kg_edges)
5. Invalidation du cache Redis
```

### Lecture (load)

```
1. Tentative de lecture depuis Redis cache
2. Si cache miss :
   a. Fetch nodes depuis Supabase
   b. Fetch edges depuis Supabase
   c. Reconstruction du graphe NetworkX
   d. Mise en cache Redis (TTL 1h)
3. Retour du graphe NetworkX
```

## Migration Supabase

### Créer les tables

Dans **Supabase SQL Editor**, exécuter `migrations/kg_tables.sql` :

```sql
-- Créer kg_nodes
CREATE TABLE kg_nodes (
    id              TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    frequency       INT  NOT NULL DEFAULT 1,
    sources         TEXT[] NOT NULL DEFAULT '{}',
    confidence_max  FLOAT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index pour performance
CREATE INDEX idx_kg_nodes_type ON kg_nodes(entity_type);
CREATE INDEX idx_kg_nodes_freq ON kg_nodes(frequency DESC);

-- Désactiver RLS pour backend
ALTER TABLE kg_nodes DISABLE ROW LEVEL SECURITY;

-- Permissions
GRANT ALL ON TABLE kg_nodes TO anon, authenticated, service_role;

-- Même chose pour kg_edges...
```

## Tests

### Test de persistence

```bash
# Test 1 - Ingest
python3 -c "
from dotenv import load_dotenv
load_dotenv()

from tools.kg_tool import ingest_text, stats

result = ingest_text('Aspirin reduces myocardial infarction', source='TEST', provider='gliner', entity_types=['DRUG','DISEASE'])
print('Après ingest:', stats())
"

# Test 2 - Restart (nouveau process)
python3 -c "
from dotenv import load_dotenv
load_dotenv()

from tools.kg_tool import stats
print('Après restart:', stats())
"
```

Si Test 2 affiche les mêmes stats que Test 1, la persistence fonctionne.

### Test du cache Redis

```bash
python3 test_redis_cache.py
```

Résultat attendu :
- ✅ Redis connecté
- 🚀 Speedup > 10x
- ✅ Cache invalidé après ingest

## Troubleshooting

### Erreur : "Supabase not configured"

**Cause** : Variables d'environnement manquantes

**Solution** :
```bash
# Vérifier .env
cat .env | grep SUPABASE

# Ajouter si manquant
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key
```

### Erreur : "Could not find table in schema cache"

**Cause** : Mauvaise base de données Supabase ou tables non créées

**Solution** :
1. Vérifier que `SUPABASE_URL` pointe vers la bonne base (med-assist)
2. Exécuter `migrations/kg_tables.sql` dans Supabase SQL Editor
3. Vérifier que les tables existent dans Table Editor

### Erreur : "Redis not configured"

**Cause** : Redis non démarré ou `REDIS_URL` manquant

**Solution** :
```bash
# Démarrer Redis
sudo systemctl start redis-server

# Ou Docker
docker run -d -p 6379:6379 redis:alpine

# Ajouter dans .env
REDIS_URL=redis://localhost:6379/0
```

### Cache ne s'invalide pas

**Cause** : Redis non accessible pendant l'écriture

**Solution** :
```python
# Invalider manuellement
from storage.kg_cache_redis import invalidate_graph_cache
invalidate_graph_cache()
```

## Maintenance

### Nettoyer le cache Redis

```python
from storage.kg_cache_redis import invalidate_graph_cache
invalidate_graph_cache()
```

### Reconstruire le graphe depuis Supabase

```python
from kg.store import load_graph
from storage.kg_cache_redis import invalidate_graph_cache

# Invalider le cache
invalidate_graph_cache()

# Recharger depuis Supabase
G = load_graph()
```

### Statistiques du cache

```python
from storage.kg_cache_redis import get_cache_stats

stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2f}%")
print(f"Hits: {stats['keyspace_hits']}")
print(f"Misses: {stats['keyspace_misses']}")
```

## Roadmap

- [ ] Support de relations typées (TREATS, CAUSES, etc.)
- [ ] Extraction de relations via LLM
- [ ] Visualisation du graphe (D3.js)
- [ ] Export GraphML/GEXF
- [ ] Recherche sémantique avec embeddings
- [ ] Clustering de communautés
- [ ] Détection d'anomalies

## Références

- **NetworkX** : https://networkx.org/
- **Supabase** : https://supabase.com/docs
- **Redis** : https://redis.io/docs/
- **GLiNER** : https://github.com/urchade/GLiNER
