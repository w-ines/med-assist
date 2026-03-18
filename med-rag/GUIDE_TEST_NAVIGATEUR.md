# Guide de Test - RAG + Knowledge Graph (Navigateur)

## 🚀 Démarrage de l'interface

### 1. Démarrer le backend (FastAPI)

```bash
cd /home/iscpif/Documents/cnrs-agent-workspace/med-assist/med-rag
python3 main.py
```

Le backend démarre sur : **http://localhost:8000**

### 2. Démarrer le frontend (Next.js)

```bash
cd /home/iscpif/Documents/cnrs-agent-workspace/med-assist/ui-med-rag
npm run dev
```

Le frontend démarre sur : **http://localhost:3000**

### 3. Ouvrir dans le navigateur

Accéder à : **http://localhost:3000**

---

## 📝 Types de Questions pour Tester

### A. Questions RAG (Retrieval-Augmented Generation)

Ces questions testent la recherche vectorielle dans les documents uploadés.

#### 1. Questions de Résumé
```
- Résume ce document
- Quels sont les points principaux de ce document ?
- De quoi parle ce document ?
- Donne-moi un aperçu du contenu
```

#### 2. Questions Spécifiques
```
- Quels sont les effets secondaires de l'aspirine ?
- Quelle est la posologie recommandée pour [médicament] ?
- Quelles sont les contre-indications de [traitement] ?
- Comment traiter [maladie] selon ce document ?
```

#### 3. Questions de Recherche
```
- Trouve toutes les mentions de "hypertension" dans le document
- Quels médicaments sont mentionnés pour le diabète ?
- Liste les symptômes décrits dans le document
- Quelles sont les interactions médicamenteuses mentionnées ?
```

#### 4. Questions Comparatives
```
- Quelle est la différence entre [médicament A] et [médicament B] ?
- Compare les traitements pour [maladie X] et [maladie Y]
- Quels sont les avantages et inconvénients de [traitement] ?
```

---

### B. Questions Knowledge Graph (KG)

Ces questions testent les relations entre entités médicales dans le graphe de connaissances.

#### 1. Questions sur les Relations
```
- Quelles sont les relations entre aspirine et infarctus du myocarde ?
- Quels médicaments sont liés à l'hypertension ?
- Quelles maladies sont associées à [symptôme] ?
- Montre-moi les connexions entre [entité A] et [entité B]
```

#### 2. Questions sur les Entités
```
- Quelles sont les entités médicales principales dans le document ?
- Liste tous les médicaments mentionnés
- Quelles maladies sont discutées ?
- Quels sont les symptômes identifiés ?
```

#### 3. Questions de Découverte
```
- Quelles sont les co-occurrences fréquentes avec [médicament] ?
- Quels concepts sont souvent mentionnés ensemble ?
- Y a-t-il des relations inattendues dans le graphe ?
```

---

### C. Questions Hybrides (RAG + KG)

Ces questions bénéficient de l'enrichissement KG pour améliorer la pertinence.

#### 1. Questions Contextuelles
```
- Explique le rôle de l'aspirine dans les maladies cardiovasculaires
- Comment [médicament] affecte-t-il [système corporel] ?
- Quels sont les mécanismes d'action de [traitement] ?
```

#### 2. Questions de Synthèse
```
- Donne-moi une vue d'ensemble des traitements pour [maladie]
- Quelles sont toutes les informations disponibles sur [médicament] ?
- Synthétise les connaissances sur [condition médicale]
```

#### 3. Questions Multi-Documents
```
- Compare les informations sur [sujet] à travers tous les documents
- Y a-t-il des contradictions dans les documents uploadés ?
- Quelles sont les recommandations communes dans tous les documents ?
```

---

## 🧪 Scénarios de Test Complets

### Scénario 1 : Upload + Questions RAG

**Étapes** :
1. Uploader un document médical (PDF)
2. Attendre l'indexation (quelques secondes)
3. Poser des questions :

```
Question 1: "Résume ce document en 3 points principaux"
→ Teste : Retrieval global + synthèse

Question 2: "Quels sont les effets secondaires mentionnés ?"
→ Teste : Retrieval ciblé + extraction d'info

Question 3: "Quelle est la posologie recommandée ?"
→ Teste : Retrieval précis + citation de source
```

**Résultat attendu** :
- ✅ Réponses basées sur le document uploadé
- ✅ Citations de sources [Source 1], [Source 2]
- ✅ Pas d'hallucinations (si info absente, le dire)

---

### Scénario 2 : Test du Knowledge Graph

**Prérequis** : Avoir ingéré du texte dans le KG

```bash
# Peupler le KG (si vide)
python3 -c "
from tools.kg_tool import ingest_text
ingest_text('Aspirin is used to treat cardiovascular diseases and reduce the risk of myocardial infarction.')
ingest_text('Hypertension is a risk factor for cardiovascular diseases and stroke.')
"
```

**Questions** :
```
Question 1: "Quelles entités médicales sont dans le graphe ?"
→ Teste : Accès au KG

Question 2: "Quelles sont les relations entre aspirine et maladies cardiovasculaires ?"
→ Teste : Requête de relations KG

Question 3: "Quels concepts sont liés à l'hypertension ?"
→ Teste : Exploration du graphe
```

**Résultat attendu** :
- ✅ Liste des entités (DRUG, DISEASE, etc.)
- ✅ Relations avec poids/fréquence
- ✅ Contexte enrichi par le KG

---

### Scénario 3 : Test Hybride RAG + KG

**Étapes** :
1. Uploader un document sur l'aspirine
2. S'assurer que le KG contient des entités liées
3. Poser une question qui bénéficie du KG :

```
Question: "Explique comment l'aspirine prévient l'infarctus du myocarde"

Résultat attendu :
- Informations du document (RAG)
- Relations du KG (aspirine → cardiovascular disease → myocardial infarction)
- Score hybride élevé pour les chunks pertinents
- Contexte enrichi : "Related entities: Aspirin, Myocardial Infarction, Cardiovascular Disease"
```

---

## 🔍 Vérification des Résultats

### Indicateurs de bon fonctionnement RAG

✅ **Citations de sources** : `[Source 1]`, `[Source 2]`  
✅ **Pas d'hallucinations** : Si info absente → "Je n'ai pas trouvé cette information"  
✅ **Pertinence** : Réponses basées sur le contenu uploadé  
✅ **Chunks affichés** : Métadonnées visibles (filename, chunk_index)

### Indicateurs de bon fonctionnement KG

✅ **Entités extraites** : DRUG, DISEASE, SYMPTOM, etc.  
✅ **Relations affichées** : "3 relationships found in Knowledge Graph"  
✅ **Score KG** : `kg_score` dans les métadonnées  
✅ **Score hybride** : `hybrid_score = (vector × 0.7) + (KG × 0.3)`

### Indicateurs d'enrichissement KG

✅ **Contexte enrichi** : "Related entities: X, Y, Z"  
✅ **Re-ranking** : Documents avec entités KG sont mieux classés  
✅ **Découverte de liens** : Relations non explicites dans le document

---

## 🐛 Dépannage

### Problème : "No documents found"

**Cause** : Aucun document indexé  
**Solution** :
```bash
# Vérifier les documents indexés
python3 -c "
from storage.supabase_client import get_supabase_client
supabase = get_supabase_client()
result = supabase.table('documents').select('*').limit(5).execute()
print(f'Documents: {len(result.data)}')
"
```

### Problème : "KG is empty"

**Cause** : Knowledge Graph non peuplé  
**Solution** :
```bash
# Peupler le KG
python3 -c "
from tools.kg_tool import ingest_text, stats
ingest_text('Sample medical text with entities...')
print(stats())
"
```

### Problème : "Low relevance warning"

**Cause** : Scores de similarité faibles  
**Solution** :
- Reformuler la question
- Vérifier que le document contient l'information
- Augmenter `top_k` pour plus de résultats

### Problème : "KG enrichment not working"

**Vérifications** :
```bash
# 1. Vérifier que le KG contient des données
python3 -c "from tools.kg_tool import stats; print(stats())"

# 2. Vérifier que GLiNER fonctionne
python3 -c "
from ner.gliner_extractor import extract_entities_gliner
result = extract_entities_gliner('Aspirin treats cardiovascular disease')
print(result)
"

# 3. Vérifier le cache Redis KG
python3 test_redis_cache.py
```

---

## 📊 Exemples de Questions par Domaine

### Cardiologie
```
- Quels sont les traitements de l'insuffisance cardiaque ?
- Comment prévenir l'infarctus du myocarde ?
- Quels sont les facteurs de risque cardiovasculaires ?
- Quelle est la différence entre angine et infarctus ?
```

### Diabétologie
```
- Quels sont les types de diabète ?
- Comment gérer le diabète de type 2 ?
- Quels sont les médicaments antidiabétiques ?
- Quelles sont les complications du diabète ?
```

### Pharmacologie
```
- Quelles sont les interactions médicamenteuses de [médicament] ?
- Quelle est la pharmacocinétique de [médicament] ?
- Quels sont les effets indésirables de [classe thérapeutique] ?
- Comment ajuster la posologie chez le patient âgé ?
```

### Oncologie
```
- Quels sont les protocoles de chimiothérapie pour [cancer] ?
- Quels sont les effets secondaires de [traitement oncologique] ?
- Comment gérer les nausées induites par la chimiothérapie ?
- Quelles sont les thérapies ciblées disponibles ?
```

---

## 🎯 Métriques de Performance

### Temps de réponse attendus

| Opération | Temps | Notes |
|-----------|-------|-------|
| Upload + indexation | 5-15s | Dépend de la taille du PDF |
| Question RAG simple | 2-4s | Vector search + LLM |
| Question RAG + KG | 3-5s | +1s pour enrichissement KG |
| Question KG seul | 1-2s | Graphe en cache Redis |

### Qualité des réponses

**Excellente** (score > 0.8) :
- Réponse précise et complète
- Citations correctes
- Contexte KG pertinent

**Bonne** (score 0.6-0.8) :
- Réponse correcte mais incomplète
- Quelques citations manquantes
- Enrichissement KG partiel

**Faible** (score < 0.6) :
- Réponse générique ou hors sujet
- Pas de citations
- Warning "Low relevance"

---

## 🔧 Configuration Recommandée

### Pour tester le RAG

```bash
# .env
EMBEDDINGS_PROVIDER=openrouter
EMBEDDINGS_MODEL=text-embedding-3-small
OPEN_ROUTER_KEY=your-key
```

### Pour tester le KG

```bash
# .env
REDIS_URL=redis://localhost:6379/0
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key
```

### Pour tester RAG + KG

```bash
# Activer l'enrichissement KG dans le retriever
enable_kg_enrichment=True
kg_weight=0.3  # 30% du score vient du KG
```

---

## 📝 Checklist de Test

Avant de tester, vérifier :

- [ ] Backend FastAPI démarré (port 8000)
- [ ] Frontend Next.js démarré (port 3000)
- [ ] Redis démarré (pour cache KG)
- [ ] Supabase configuré (.env)
- [ ] Au moins 1 document uploadé et indexé
- [ ] KG peuplé avec quelques entités
- [ ] GLiNER disponible (NER)

Pendant le test :

- [ ] Upload d'un document fonctionne
- [ ] Questions RAG retournent des réponses
- [ ] Citations de sources présentes
- [ ] Enrichissement KG visible dans les métadonnées
- [ ] Pas d'erreurs dans la console backend
- [ ] Temps de réponse acceptable (< 5s)

---

## 🎓 Ressources

- **README_RAG.md** : Documentation complète du système RAG
- **README_KG.md** : Documentation du Knowledge Graph
- **test_rag_kg.py** : Script de test automatisé
- **test_redis_cache.py** : Test du cache Redis KG

---

Bon test ! 🚀
