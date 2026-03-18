# 🔬 Projet : Veille Biomédicale Intelligente avec IA

**Nom du Projet :** medAssist - Intelligent Biomedical Intelligence Agent  
**Version :** 1.0  
**Date :** Février 2026  
**Auteur :** [Ton Nom]

---

## 📋 Table des Matières

1. [Vue d'Ensemble du Projet](#vue-densemble-du-projet)
2. [Objectifs & Proposition de Valeur](#objectifs--proposition-de-valeur)
3. [Public Cible & Use Cases](#public-cible--use-cases)
4. [Architecture Technique](#architecture-technique)
5. [Fonctionnalités Détaillées](#fonctionnalités-détaillées)
6. [Stack Technologique](#stack-technologique)
7. [Roadmap de Développement](#roadmap-de-développement)
8. [Plan de Validation](#plan-de-validation)
9. [Stratégie de Déploiement](#stratégie-de-déploiement)
10. [Monétisation & Business Model](#monétisation--business-model)
11. [Risques & Mitigations](#risques--mitigations)
12. [Extensions Futures](#extensions-futures)

---

## 1. Vue d'Ensemble du Projet

### 🎯 Concept Central

**medAssist** est un agent IA de veille biomédicale automatisée qui :
- Surveille quotidiennement PubMed selon les intérêts de l'utilisateur
- Extrait automatiquement les entités médicales (maladies, gènes, médicaments) avec OpenMed
- Détecte les tendances émergentes et signaux faibles
- Génère des résumés intelligents et actionnables
- Stocke et indexe la connaissance dans un vector store pour recherche sémantique

### 💡 Innovation Clé

Contrairement aux alertes PubMed classiques (simples emails de nouveaux articles), medAssist :
- ✅ **Comprend** le contenu (NER avec OpenMed)
- ✅ **Analyse** les tendances (clustering, détection d'émergence)
- ✅ **Synthétise** l'information (LLM summarization)
- ✅ **Mémorise** dans un knowledge graph interrogeable

### 🚀 Positionnement

**Entre :**
- Alertes PubMed (trop brutes, pas d'analyse)
- Services payants comme F1000, UpToDate (chers, fermés)

**medAssist = Open-source, intelligent, personnalisable**

---

## 2. Objectifs & Proposition de Valeur

### 🎯 Objectifs Principaux

#### Court Terme (3 mois)
1. ✅ Construire MVP fonctionnel avec 5 features core
2. ✅ Tester avec 10-15 chercheurs/doctorants
3. ✅ Valider l'approche technique (OpenMed + RAG)
4. ✅ Open-sourcer sur GitHub avec documentation

#### Moyen Terme (6-12 mois)
1. ✅ Atteindre 100 utilisateurs actifs
2. ✅ Intégrer 3-5 features avancées (citation network, comparative analysis)
3. ✅ Déployer version hosted (SaaS light)
4. ✅ Publier article/blog post technique

#### Long Terme (12-24 mois)
1. ✅ 1000+ utilisateurs (chercheurs, startups biotech)
2. ✅ Intégration avec outils existants (Zotero, Mendeley)
3. ✅ Marketplace de "intelligence agents" custom
4. ✅ Évolution vers Clinical Decision Support

### 💎 Proposition de Valeur

#### Pour Doctorants/Chercheurs
> "Économise 5-10h/semaine de veille bibliographique manuelle. Reste à jour dans ton domaine sans submerger ta boîte mail."

**Avant medAssist :**
- 📧 50-100 emails d'alerte PubMed/semaine
- 📚 2-3h pour trier et lire abstracts pertinents
- 😰 Peur de manquer THE article important
- 🔍 Recherches manuelles répétitives

**Après medAssist :**
- 📊 1 digest hebdomadaire structuré (15min lecture)
- 🎯 Top 10 articles pertinents pré-sélectionnés
- 📈 Tendances émergentes détectées automatiquement
- 🔔 Alertes sur signaux critiques (breakthrough, concurrent)

#### Pour Startups Biotech
> "Intelligence compétitive automatisée. Surveille concurrents, brevets, essais cliniques."

**Valeur :**
- Veille concurrentielle H24
- Détection early de nouvelles technologies
- Analyse de landscape (qui travaille sur quoi?)
- ROI : évite 1 consultant à 5k€/mois

#### Pour Médecins-Chercheurs
> "Reste à jour sur ta spécialité sans sacrifier ton temps clinique."

**Valeur :**
- Digest personnalisé selon spécialité
- Filtrage haute qualité (RCT, meta-analyses prioritaires)
- Lien direct vers full-text si accès institutionnel
- Integration avec CPD/CME tracking

---

## 3. Public Cible & Use Cases

### 👥 Personas Détaillés

#### Persona 1 : Marie, Doctorante en Biologie Moléculaire
**Profil :**
- 26 ans, 2ème année de thèse
- Sujet : Résistance aux antibiotiques chez E. coli
- Université Paris-Saclay

**Besoins :**
- Suivre 3-4 topics précis (NDM-1, OXA-48, carbapenemases)
- Identifier nouveaux mécanismes de résistance
- Surveiller équipes concurrentes
- Préparer revue de littérature

**Usage medAssist :**
```yaml
topics:
  - "NDM-1 carbapenemase"
  - "OXA-48 beta-lactamase"
  - "antibiotic resistance E. coli"
  - "efflux pump mechanisms"
frequency: daily
filters:
  - journals: ["Nature Microbiology", "mBio", "AAC"]
  - publication_types: ["Research Article", "Review"]
  - exclude_species: ["human", "mouse"]  # Focus bacteria
```

**Résultat attendu :**
- Email digest chaque matin 9h
- 5-15 nouveaux articles/jour
- Clustering par mécanisme (efflux, enzymatic, permeability)
- Alert si concurrent publie dans Nature/Science

---

#### Persona 2 : Dr. Chen, Oncologue-Chercheur
**Profil :**
- 42 ans, MD-PhD
- Chef de service oncologie médicale
- 70% clinique, 30% recherche
- Focus : Immunothérapie cancer du poumon

**Besoins :**
- Rester à jour malgré charge clinique
- Identifier nouveaux essais cliniques
- Préparer conférences/formations
- Anticiper questions patients ("j'ai lu sur internet que...")

**Usage medAssist :**
```yaml
topics:
  - "lung cancer immunotherapy"
  - "PD-1 PD-L1 inhibitors"
  - "NSCLC targeted therapy"
frequency: weekly  # Pas le temps quotidien
filters:
  - evidence_level: ["Meta-Analysis", "RCT", "Phase III"]
  - exclude: ["preclinical", "mouse models"]
  - priority_journals: ["JCO", "Lancet Oncology", "NEJM"]
```

**Résultat attendu :**
- Email digest vendredi 18h (lecture weekend)
- Top 10 articles must-read
- Section "Clinical Trials" séparée
- Résumés en français si possible

---

#### Persona 3 : BioPharm Inc, Startup Thérapie Génique
**Profil :**
- 15 employés, Serie A (5M€)
- Développe AAV vectors pour maladies rares
- Besoin veille concurrentielle + state-of-the-art

**Besoins :**
- Surveiller concurrents (qui publie quoi)
- Détecter nouvelles technologies (CRISPR variants)
- Identifier opportunités partenariats
- Préparer pitchs investors (market landscape)

**Usage medAssist :**
```yaml
topics:
  - "AAV gene therapy"
  - "adeno-associated virus vectors"
  - "CRISPR base editing"
  - "Duchenne muscular dystrophy treatment"
competitors:
  - "Spark Therapeutics"
  - "Sarepta Therapeutics"
  - "Solid Biosciences"
frequency: daily
alerts:
  - competitor_publication: immediate
  - fda_approval: immediate
  - clinical_trial_start: daily
```

**Résultat attendu :**
- Dashboard Slack intégré
- Alertes push si concurrent publie
- Weekly report pour CEO/investors
- Comparative analysis (nous vs. market)

---

### 📊 Use Cases Détaillés

#### Use Case 1 : Veille Thématique Standard
**Input :**
```python
agent.track_topics([
    "CRISPR gene editing",
    "mRNA vaccine technology"
])
```

**Processus :**
1. Recherche PubMed quotidienne (nouveaux articles)
2. Extraction entités (gènes, maladies, médicaments)
3. Clustering thématique
4. Génération résumé
5. Email digest

**Output :**
```markdown
# Digest Quotidien - 24 Février 2026

## CRISPR Gene Editing (23 nouveaux articles)

### Tendances Émergentes
- Base editing: ↑ 45% mentions vs. mois dernier
- CRISPR-Cas12: 12 nouvelles applications

### Top Articles
1. **Nature Biotechnology** - "Improved base editor with minimal off-target"
   - Entities: CRISPR, adenine base editor, off-target effects
   - PMID: 38234567

2. **Cell** - "In vivo CRISPR screening identifies cancer dependencies"
   - Entities: cancer, CRISPR screen, genetic dependencies
   - PMID: 38234568

### Signaux à Surveiller
⚠️ 3 articles mentionnent toxicité hépatique post-AAV delivery
```

---

#### Use Case 2 : Détection de Tendances
**Input :**
```python
trends = agent.detect_emerging_trends(
    topic="immunotherapy",
    time_window=90  # last 3 months
)
```

**Output :**
```json
{
  "emerging_entities": [
    {
      "entity": "bispecific T-cell engagers",
      "type": "TREATMENT",
      "mention_increase": "+127%",
      "articles": 45,
      "key_papers": ["PMID:123", "PMID:456"]
    },
    {
      "entity": "tumor-infiltrating lymphocytes",
      "type": "BIOMARKER",
      "mention_increase": "+89%",
      "articles": 32
    }
  ],
  "declining_entities": [
    {
      "entity": "CTLA-4 inhibitors",
      "mention_decrease": "-23%",
      "reason": "Market saturation, focus shifts to novel targets"
    }
  ]
}
```

---

#### Use Case 3 : Surveillance Concurrentielle
**Input :**
```python
agent.monitor_competitors([
    "Moderna Inc",
    "BioNTech SE",
    "CureVac"
], focus="mRNA therapeutics")
```

**Processus :**
1. Recherche auteur affiliations
2. Extraction co-auteurs (réseau collaborations)
3. Analyse pipeline (clinical trials mentions)
4. Détection brevets (via PubMed/Google Patents)

**Output :**
```markdown
# Competitive Intelligence Report - Week 8/2026

## Moderna Inc
- 4 nouvelles publications cette semaine
- Focus: mRNA cancer vaccines (3/4 articles)
- Collaboration détectée: MD Anderson Cancer Center
- Clinical trial NCT05432156 en Phase II (melanoma)

## BioNTech SE
- 2 publications (dont 1 Nature)
- Breakthrough: Personalized cancer vaccine platform
- 12 co-auteurs nouveaux → expansion équipe R&D?

## CureVac
- ⚠️ Aucune publication depuis 3 semaines (inhabituel)
- Analyse: possiblement pivot stratégique ou phase silencieuse pre-annonce

## Recommandations
→ Surveiller Moderna cancer vaccines (competitive threat)
→ Analyser BioNTech patent applications (defensive strategy)
```

---

## 4. Architecture Technique

### 🏗️ Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  Web Dashboard  │  Email Digest  │  API  │  Slack/Discord  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  • Task Scheduler (Celery/APScheduler)                      │
│  • User Profile Manager                                     │
│  • Notification Manager                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  PubMed Search  →  OpenMed NER  →  Clustering  →  LLM      │
│   (Biopython)    (Entity Extract)  (Themes)    (Summary)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Vector Store   │  Relational DB  │  Object Storage        │
│  (Chroma/Pinecone) (PostgreSQL)   (S3/MinIO)              │
│  Embeddings     │  Metadata       │  Full-text PDFs        │
└─────────────────────────────────────────────────────────────┘
```

### 🔧 Composants Détaillés

#### A. PubMed Search Engine
```python
class PubMedSearchEngine:
    """Interface PubMed avec rate limiting et caching"""
    
    def __init__(self, email: str, api_key: str = None):
        self.email = email
        self.api_key = api_key  # NCBI API key (10 req/s vs 3 req/s)
        self.cache = Redis()  # Cache résultats
    
    def search(
        self,
        query: str,
        date_range: int = 30,
        max_results: int = 100,
        filters: dict = None
    ) -> List[Article]:
        """
        Search PubMed with advanced filters
        
        Filters:
        - publication_types: ["Clinical Trial", "Meta-Analysis"]
        - journals: ["Nature", "Science", "Cell"]
        - languages: ["eng"]
        - species: ["Humans"]
        """
        # Build Entrez query
        entrez_query = self._build_query(query, date_range, filters)
        
        # Check cache
        cache_key = f"pubmed:{hash(entrez_query)}"
        if cached := self.cache.get(cache_key):
            return cached
        
        # Search
        handle = Entrez.esearch(
            db="pubmed",
            term=entrez_query,
            retmax=max_results,
            sort="relevance",
            api_key=self.api_key
        )
        
        results = Entrez.read(handle)
        pmids = results['IdList']
        
        # Fetch full records
        articles = self._fetch_articles(pmids)
        
        # Cache for 24h
        self.cache.setex(cache_key, 86400, articles)
        
        return articles
    
    def _build_query(self, base_query: str, days: int, filters: dict) -> str:
        """Construct advanced Entrez query"""
        parts = [base_query]
        
        # Date filter
        end = datetime.now()
        start = end - timedelta(days=days)
        parts.append(f"({start:%Y/%m/%d}:{end:%Y/%m/%d}[pdat])")
        
        # Publication type
        if pub_types := filters.get('publication_types'):
            type_query = " OR ".join([f"{t}[ptyp]" for t in pub_types])
            parts.append(f"({type_query})")
        
        # Journals
        if journals := filters.get('journals'):
            journal_query = " OR ".join([f"{j}[jour]" for j in journals])
            parts.append(f"({journal_query})")
        
        # Language
        if lang := filters.get('language'):
            parts.append(f"{lang}[lang]")
        
        return " AND ".join(parts)
    
    def _fetch_articles(self, pmids: List[str]) -> List[Article]:
        """Fetch full article metadata"""
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids,
            rettype="xml",
            api_key=self.api_key
        )
        
        records = Entrez.read(handle)
        
        articles = []
        for record in records['PubmedArticle']:
            article = self._parse_record(record)
            articles.append(article)
        
        return articles
```

#### B. OpenMed NER Integration
```python
class MedicalEntityExtractor:
    """OpenMed NER wrapper avec caching et batch processing"""
    
    def __init__(self):
        from openmed import analyze_text, batch_process, OpenMedConfig
        
        self.config = OpenMedConfig(
            use_medical_tokenizer=True,
            confidence_threshold=0.7,
            group_entities=True
        )
        
        self.models = {
            'disease': 'disease_detection_superclinical',
            'drug': 'pharma_detection_superclinical',
            'gene': 'genomic_detection',
            'anatomy': 'anatomy_detection'
        }
    
    def extract_all(self, text: str) -> Dict[str, List[Entity]]:
        """Extract all entity types from text"""
        all_entities = {}
        
        for entity_type, model_name in self.models.items():
            entities = analyze_text(
                text,
                model_name=model_name,
                config=self.config
            )
            
            all_entities[entity_type] = entities
        
        return all_entities
    
    def extract_batch(self, articles: List[Article]) -> List[Dict]:
        """Batch process multiple articles efficiently"""
        from openmed import batch_process
        
        # Prepare inputs
        texts = [a.title + " " + a.abstract for a in articles]
        
        # Batch process (more efficient than loop)
        results = []
        for model_name in self.models.values():
            batch_results = batch_process(
                texts,
                model_name=model_name,
                config=self.config
            )
            results.append(batch_results)
        
        # Merge results
        merged = self._merge_batch_results(results, articles)
        return merged
    
    def normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize entity names (synonyms, abbreviations)"""
        # TODO: Link to UMLS/MeSH for standardization
        # Example: "diabetes mellitus type 2" → "Type 2 Diabetes"
        pass
```

#### C. Trend Detection Engine
```python
class TrendDetector:
    """Detect emerging topics and entities"""
    
    def __init__(self, vector_store, lookback_days: int = 90):
        self.store = vector_store
        self.lookback = lookback_days
    
    def detect_emerging(
        self,
        topic: str,
        min_growth: float = 0.5  # 50% increase
    ) -> List[Trend]:
        """
        Detect emerging entities in a topic
        
        Algorithm:
        1. Get entity counts for current period (last 30 days)
        2. Get entity counts for previous period (30-60 days ago)
        3. Calculate growth rate
        4. Flag entities with growth > threshold
        """
        # Current period
        current = self._get_entity_counts(
            topic,
            days=30,
            offset=0
        )
        
        # Previous period
        previous = self._get_entity_counts(
            topic,
            days=30,
            offset=30
        )
        
        # Calculate trends
        trends = []
        for entity, current_count in current.items():
            prev_count = previous.get(entity, 0)
            
            if prev_count > 0:
                growth = (current_count - prev_count) / prev_count
                
                if growth > min_growth:
                    trends.append(Trend(
                        entity=entity,
                        current_count=current_count,
                        previous_count=prev_count,
                        growth_rate=growth,
                        confidence=self._calculate_confidence(
                            current_count, prev_count
                        )
                    ))
            elif current_count > 5:  # New entity with significant mentions
                trends.append(Trend(
                    entity=entity,
                    current_count=current_count,
                    previous_count=0,
                    growth_rate=float('inf'),
                    confidence=0.8,
                    is_new=True
                ))
        
        return sorted(trends, key=lambda t: t.growth_rate, reverse=True)
    
    def _calculate_confidence(self, current: int, previous: int) -> float:
        """Statistical confidence in trend (avoid noise)"""
        # Simple heuristic: higher counts = higher confidence
        total = current + previous
        
        if total < 5:
            return 0.3  # Low confidence (small sample)
        elif total < 20:
            return 0.6
        else:
            return 0.9  # High confidence
```

#### D. LLM Summarizer
```python
class IntelligenceSummarizer:
    """Generate human-readable summaries with LLM"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model
    
    def generate_digest(
        self,
        articles: List[Article],
        themes: Dict[str, List[Article]],
        trends: List[Trend],
        user_context: str = None
    ) -> str:
        """Generate executive summary digest"""
        
        prompt = self._build_digest_prompt(
            articles, themes, trends, user_context
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,  # More focused, less creative
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.content[0].text
    
    def _build_digest_prompt(
        self,
        articles: List[Article],
        themes: Dict,
        trends: List[Trend],
        context: str
    ) -> str:
        """Construct detailed prompt for LLM"""
        
        return f"""
Tu es un expert en veille scientifique biomédicale. Génère un résumé exécutif structuré.

CONTEXTE UTILISATEUR:
{context or "Chercheur en biologie moléculaire"}

DONNÉES:
- {len(articles)} nouveaux articles cette semaine
- {len(themes)} thèmes principaux identifiés
- {len(trends)} tendances émergentes détectées

THÈMES IDENTIFIÉS:
{self._format_themes(themes)}

TENDANCES ÉMERGENTES:
{self._format_trends(trends)}

TOP ARTICLES:
{self._format_top_articles(articles[:10])}

INSTRUCTIONS:
1. Commence par un résumé exécutif (2-3 phrases)
2. Décris les 3 thèmes principaux avec insights clés
3. Highlight les 2-3 tendances les plus significatives
4. Liste top 5 articles must-read avec PMID
5. Termine par 2-3 points d'action ou opportunités

FORMAT:
- Utilise des bullet points pour lisibilité
- Inclus PMID pour chaque article mentionné
- Garde un ton professionnel mais accessible
- Longueur: ~500 mots

GÉNÈRE LE RÉSUMÉ:
"""
```

---

## 5. Fonctionnalités Détaillées

### 🎯 Features MVP (Semaines 1-4)

#### Feature 1 : Topic Tracking
**Description :** Suivi automatique de 1-5 topics d'intérêt

**User Story :**
> "En tant que doctorant, je veux suivre 3 topics liés à ma thèse pour recevoir un digest quotidien des nouveaux articles pertinents."

**Spécifications :**
- Input: Liste de keywords/phrases (max 5)
- Fréquence: Daily, Weekly, Custom
- Filtres: Publication types, journals, date range
- Output: Email digest HTML formaté

**Critères d'Acceptation :**
- [ ] Utilisateur peut ajouter/modifier topics via UI
- [ ] Digest envoyé à l'heure configurée
- [ ] Digest contient 5-20 articles pertinents
- [ ] Chaque article a: titre, abstract, PMID, journal

**Exemple Configuration :**
```yaml
user_id: marie_dupont
topics:
  - query: "CRISPR gene editing"
    filters:
      publication_types: ["Research Article", "Review"]
      date_range: 7  # last 7 days
  - query: "mRNA vaccine"
    filters:
      journals: ["Nature", "Science", "Cell"]
frequency: daily
delivery_time: "08:00"
timezone: "Europe/Paris"
```

---

#### Feature 2 : Entity Extraction & Tagging
**Description :** Extraction automatique entités médicales avec OpenMed

**User Story :**
> "En tant que chercheur, je veux voir les maladies, gènes et médicaments mentionnés dans les articles pour identifier rapidement les concepts clés."

**Spécifications :**
- Types d'entités: DISEASE, DRUG, GENE, ANATOMY, SPECIES
- Confidence threshold: 0.7 (production)
- Affichage: Tags colorés par type
- Stockage: Indexé dans vector store

**Critères d'Acceptation :**
- [ ] Chaque article a entités extraites
- [ ] Entités affichées avec couleurs distinctes
- [ ] Click sur entité → recherche articles similaires
- [ ] Export entités en CSV/JSON

**Exemple Output :**
```json
{
  "pmid": "38234567",
  "title": "CRISPR-Cas9 治療...",
  "entities": {
    "DISEASE": [
      {"text": "Duchenne muscular dystrophy", "confidence": 0.95},
      {"text": "cardiomyopathy", "confidence": 0.88}
    ],
    "GENE": [
      {"text": "DMD", "confidence": 0.97},
      {"text": "dystrophin", "confidence": 0.93}
    ],
    "TREATMENT": [
      {"text": "CRISPR-Cas9", "confidence": 0.99},
      {"text": "exon skipping", "confidence": 0.85}
    ]
  }
}
```

---

#### Feature 3 : Thematic Clustering
**Description :** Regroupement automatique des articles par thème

**User Story :**
> "En tant qu'utilisateur, je veux que les articles soient organisés par thème pour comprendre rapidement les différents angles de recherche."

**Spécifications :**
- Algorithme: DBSCAN ou K-means sur embeddings d'entités
- Nombre de clusters: Auto-détecté (3-8 typiquement)
- Labels: Générés par LLM à partir des entités dominantes
- Visualisation: Card-based UI

**Critères d'Acceptation :**
- [ ] Articles groupés en 3-8 thèmes cohérents
- [ ] Chaque thème a un label explicite
- [ ] Utilisateur peut voir tous articles d'un thème
- [ ] Thèmes ordonnés par taille (nb d'articles)

**Exemple Output :**
```
Thème 1: "Mécanismes de Résistance aux Antibiotiques" (12 articles)
- Efflux pumps (5 articles)
- Beta-lactamase variants (4 articles)
- Membrane permeability (3 articles)

Thème 2: "Nouvelles Molécules Antibiotiques" (8 articles)
- Peptides antimicrobiens (4 articles)
- Inhibiteurs beta-lactamase (4 articles)

Thème 3: "Épidémiologie de la Résistance" (6 articles)
- Surveillance hospitalière (3 articles)
- One Health approach (3 articles)
```

---

#### Feature 4 : Trend Detection
**Description :** Détection automatique d'entités émergentes

**User Story :**
> "En tant que chercheur, je veux être alerté quand un nouveau gène/médicament/concept émerge dans mon domaine."

**Spécifications :**
- Période de comparaison: 30 jours courants vs 30 jours précédents
- Seuil de croissance: +50% mentions minimum
- Confidence: Filtre bruit statistique (min 5 mentions)
- Alertes: Email immédiat si croissance >100%

**Critères d'Acceptation :**
- [ ] Tendances calculées quotidiennement
- [ ] Affichage croissance en %
- [ ] Lien vers articles supportant la tendance
- [ ] Historique des tendances sur 90 jours

**Exemple Output :**
```
🔥 TENDANCES ÉMERGENTES - Immunothérapie

1. "CAR-macrophages" 
   ↑ +180% (2 mentions → 12 mentions)
   Confidence: 0.85
   Articles clés: PMID:123, PMID:456

2. "Bispecific antibodies"
   ↑ +95% (10 mentions → 20 mentions)
   Confidence: 0.92
   
⚠️ SIGNAUX FAIBLES:
- "STING agonists" ↑ +60% (3 → 5 mentions)
  → À surveiller le mois prochain
```

---

#### Feature 5 : LLM Summary Generation
**Description :** Génération de résumés intelligents multi-articles

**User Story :**
> "En tant qu'utilisateur pressé, je veux un résumé exécutif de 2-3 paragraphes qui synthétise l'essentiel."

**Spécifications :**
- Longueur: 300-500 mots
- Structure: Intro → Key findings → Implications
- Citations: PMID mentionnés inline
- Tone: Professionnel mais accessible

**Critères d'Acceptation :**
- [ ] Résumé généré en <10 secondes
- [ ] Mentionne 3-5 articles les plus importants
- [ ] Identifie consensus vs controverses
- [ ] Suggestions d'actions/opportunités

**Exemple Output :**
```markdown
# Résumé Hebdomadaire - Immunothérapie Cancer

Cette semaine a vu une convergence vers les thérapies cellulaires 
de nouvelle génération, avec 23 publications explorant des alternatives 
aux CAR-T classiques.

## Découvertes Clés

Trois études majeures (PMID:123, PMID:456, PMID:789) démontrent 
l'efficacité des CAR-macrophages dans les tumeurs solides, un 
domaine où les CAR-T peinent. Les résultats précliniques montrent 
une pénétration tumorale supérieure de 3-4x vs CAR-T.

Parallèlement, les anticorps bispécifiques continuent leur montée 
en puissance avec 2 essais Phase III positifs dans les lymphomes B 
(PMID:234, PMID:567). Ces données renforcent leur position comme 
alternative off-the-shelf aux thérapies autologues.

## Implications

→ CAR-macrophages: Opportunité pour tumeurs solides (poumon, sein)
→ Anticorps bispécifiques: Compétition directe avec CAR-T
→ À surveiller: Combinaisons CAR-T + checkpoint inhibitors (3 essais débutent)
```

---

### 🚀 Features Avancées (Post-MVP)

#### Feature 6 : Citation Network Analysis
**Description :** Visualisation réseau de citations entre articles

**Valeur :**
- Identifier articles "hub" (très cités)
- Découvrir communautés de recherche
- Tracer généalogie d'une idée

**Implémentation :**
```python
# Via PubMed E-utilities LinkOut
def build_citation_network(pmids: List[str]) -> Graph:
    """Construit graphe de citations"""
    G = nx.DiGraph()
    
    for pmid in pmids:
        # Get articles citant ce PMID
        citing = get_citing_articles(pmid)
        
        # Get articles cités par ce PMID
        cited = get_cited_articles(pmid)
        
        G.add_edges_from([(pmid, c) for c in citing])
        G.add_edges_from([(c, pmid) for c in cited])
    
    return G

# Analyse
central_nodes = nx.betweenness_centrality(G)
communities = nx.community.louvain_communities(G)
```

---

#### Feature 7 : Comparative Analysis
**Description :** Comparaison de 2+ topics/périodes

**Use Case :**
> "Comparer traitement A vs traitement B dans la littérature"

**Output :**
```
COMPARAISON: Pembrolizumab vs Nivolumab (Cancer Poumon)

Pembrolizumab:
- 234 articles (2024-2026)
- OS médian: 12.2 mois (8 études)
- Grade 3+ AE: 18% (méta-analyse)
- Coût: $$$

Nivolumab:
- 189 articles
- OS médian: 11.8 mois (6 études)
- Grade 3+ AE: 15%
- Coût: $$

Consensus: Efficacité similaire, légère différence toxicité
Controverses: Biomarker PD-L1 seuil (1% vs 50%)
```

---

#### Feature 8 : Clinical Trial Tracker
**Description :** Détection et suivi des essais cliniques mentionnés

**Spécifications :**
- Parse abstracts pour NCT numbers
- Fetch détails via ClinicalTrials.gov API
- Track statut (recruiting, completed, etc.)
- Alert si résultats publiés

---

#### Feature 9 : Zotero/Mendeley Integration
**Description :** Export direct vers gestionnaires bibliographiques

**Features :**
- One-click export sélection articles
- Sync automatique digest → collection Zotero
- Tags = entités extraites par OpenMed

---

#### Feature 10 : Collaborative Workspaces
**Description :** Espaces partagés pour équipes

**Use Case :**
> Lab de 10 chercheurs partage veille sur projet commun

**Features :**
- Workspace partagé avec topics communs
- Annotations collaboratives
- Assign articles à membres équipe
- Discussion threads par article

---

## 6. Stack Technologique

### 🛠️ Technologies Core

```yaml
Backend:
  Language: Python 3.11+
  Framework: FastAPI (API REST)
  Task Queue: Celery + Redis
  Scheduler: APScheduler
  
Data Science:
  NER: OpenMed (Transformers)
  LLM: Anthropic Claude Sonnet 4
  Embeddings: sentence-transformers (all-MiniLM-L6-v2)
  Clustering: scikit-learn (DBSCAN, KMeans)
  
Storage:
  Vector Store: ChromaDB (dev) → Pinecone (prod)
  Relational DB: PostgreSQL 15
  Cache: Redis 7
  Object Storage: MinIO (self-hosted) / S3 (cloud)
  
Frontend:
  Framework: Streamlit (MVP) → Next.js + React (v2)
  Styling: Tailwind CSS
  Charts: Recharts / Plotly
  
Infrastructure:
  Container: Docker + Docker Compose
  Orchestration: Kubernetes (if scale)
  CI/CD: GitHub Actions
  Monitoring: Grafana + Prometheus
  Logging: ELK Stack (Elasticsearch, Logstash, Kibana)
  
External APIs:
  PubMed: NCBI E-utilities
  Clinical Trials: ClinicalTrials.gov API
  MeSH: NLM MeSH API
```

### 📦 Dependencies Python

```toml
# pyproject.toml

[tool.poetry.dependencies]
python = "^3.11"

# Core
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
pydantic = "^2.5.0"
python-dotenv = "^1.0.0"

# OpenMed & NLP
openmed = "^0.5.0"
transformers = "^4.37.0"
sentence-transformers = "^2.3.0"
biopython = "^1.83"

# LLM
anthropic = "^0.18.0"

# Data Science
numpy = "^1.26.0"
pandas = "^2.1.0"
scikit-learn = "^1.4.0"
networkx = "^3.2"

# Storage
chromadb = "^0.4.22"
psycopg2-binary = "^2.9.9"
redis = "^5.0.1"
sqlalchemy = "^2.0.25"

# Task Queue
celery = "^5.3.6"
apscheduler = "^3.10.4"

# UI
streamlit = "^1.31.0"
plotly = "^5.18.0"

# Utilities
requests = "^2.31.0"
httpx = "^0.26.0"
python-multipart = "^0.0.9"
email-validator = "^2.1.0"
```

---

## 7. Roadmap de Développement

### 📅 Timeline Détaillée (16 Semaines)

#### **Phase 1 : Foundation (Semaines 1-4)**

##### Semaine 1 : Setup & PubMed Integration
**Objectif :** Infrastructure de base + recherche PubMed fonctionnelle

**Tasks :**
- [ ] Init repo GitHub + structure projet
- [ ] Setup Docker Compose (PostgreSQL, Redis, MinIO)
- [ ] Implémenter `PubMedSearchEngine` class
- [ ] Tests unitaires recherche PubMed
- [ ] Documentation API PubMed

**Deliverables :**
```python
# Functionality
engine = PubMedSearchEngine(email="...")
articles = engine.search("CRISPR", date_range=7)
# → Returns 50-100 articles with metadata
```

**Critères de Succès :**
- ✅ Recherche PubMed avec filtres avancés
- ✅ Rate limiting respecté (3 req/s)
- ✅ Caching Redis fonctionnel
- ✅ 20+ tests passent

---

##### Semaine 2 : OpenMed NER Integration
**Objectif :** Extraction entités sur articles PubMed

**Tasks :**
- [ ] Installer OpenMed + tous modèles NER
- [ ] Wrapper `MedicalEntityExtractor` avec batch
- [ ] Pipeline: PubMed → OpenMed → Structured output
- [ ] Tests sur 100 abstracts réels

**Deliverables :**
```python
extractor = MedicalEntityExtractor()
entities = extractor.extract_all(article.abstract)
# → {DISEASE: [...], DRUG: [...], GENE: [...]}
```

**Critères de Succès :**
- ✅ Extraction 4 types entités (disease, drug, gene, anatomy)
- ✅ Batch processing 100 articles en <30s
- ✅ Confidence threshold configurable
- ✅ Entités stockées en DB

---

##### Semaine 3 : Vector Store & Storage
**Objectif :** Stockage persistant + recherche sémantique

**Tasks :**
- [ ] Setup ChromaDB collection
- [ ] Embeddings avec sentence-transformers
- [ ] Store articles + métadonnées PostgreSQL
- [ ] Index entités pour recherche rapide

**Deliverables :**
```python
# Store
store.add_articles(articles, embeddings)

# Semantic search
similar = store.search(
    "articles about CRISPR toxicity",
    top_k=10
)
```

**Critères de Succès :**
- ✅ 1000+ articles indexés
- ✅ Recherche sémantique <200ms
- ✅ Metadata queries SQL rapides
- ✅ Backup/restore fonctionnel

---

##### Semaine 4 : Clustering & Trends
**Objectif :** Analyse thématique et détection tendances

**Tasks :**
- [ ] Implémentation clustering (DBSCAN)
- [ ] Trend detection algorithm
- [ ] Label generation avec LLM
- [ ] Tests sur datasets réels

**Deliverables :**
```python
# Clustering
themes = cluster_articles(articles)
# → {theme_1: [art1, art2], theme_2: [art3, art4]}

# Trends
trends = detect_trends(topic="immunotherapy", days=30)
# → [{entity: "CAR-macrophages", growth: +180%}]
```

**Critères de Succès :**
- ✅ Clustering cohérent sur 100+ articles
- ✅ Trends détectées avec confidence scores
- ✅ Labels thèmes interprétables
- ✅ Validation manuelle sur 10 cas

---

#### **Phase 2 : Intelligence (Semaines 5-8)**

##### Semaine 5 : LLM Summarization
**Objectif :** Génération résumés multi-articles

**Tasks :**
- [ ] Integration Anthropic Claude API
- [ ] Prompt engineering pour digests
- [ ] Templates résumés (daily, weekly)
- [ ] A/B testing prompts

**Deliverables :**
```python
summarizer = IntelligenceSummarizer()
digest = summarizer.generate_digest(
    articles=articles,
    themes=themes,
    trends=trends,
    user_context="PhD student in cancer biology"
)
# → 500-word structured summary
```

**Critères de Succès :**
- ✅ Résumés générés en <10s
- ✅ Qualité validée par 3 chercheurs
- ✅ Citations PMID correctes
- ✅ Tone approprié (pro mais accessible)

---

##### Semaine 6 : Task Scheduling
**Objectif :** Automation quotidienne/hebdomadaire

**Tasks :**
- [ ] Setup Celery workers
- [ ] Scheduled tasks (daily digest, weekly report)
- [ ] User preference management
- [ ] Error handling & retries

**Deliverables :**
```python
# Celery task
@celery.task
def generate_daily_digest(user_id: str):
    user = get_user(user_id)
    articles = search_new_articles(user.topics)
    digest = create_digest(articles)
    send_email(user.email, digest)

# Schedule
schedule.every().day.at("08:00").do(
    generate_daily_digest, user_id="marie"
)
```

**Critères de Succès :**
- ✅ Tasks s'exécutent à l'heure prévue
- ✅ Retry logic si API failure
- ✅ Monitoring via Flower
- ✅ 10 users en test reçoivent digests

---

##### Semaine 7 : Email Digest System
**Objectif :** Templates email beaux et fonctionnels

**Tasks :**
- [ ] HTML email templates (Jinja2)
- [ ] Responsive design (mobile-friendly)
- [ ] Unsubscribe/preferences links
- [ ] SMTP config (SendGrid/AWS SES)

**Deliverables :**
```html
<!-- Email template -->
<html>
  <body>
    <h1>Votre Digest Quotidien - {{ date }}</h1>
    
    <section>
      <h2>🔥 Tendances Émergentes</h2>
      {% for trend in trends %}
        <div class="trend">
          <strong>{{ trend.entity }}</strong> 
          <span class="growth">↑ {{ trend.growth }}%</span>
        </div>
      {% endfor %}
    </section>
    
    <!-- More sections -->
  </body>
</html>
```

**Critères de Succès :**
- ✅ Email rendu correct sur Gmail, Outlook, Apple Mail
- ✅ Images inline (pas d'attachments)
- ✅ CTA clairs (Read more, Unsubscribe)
- ✅ Deliverability >95% (pas de spam folder)

---

##### Semaine 8 : User Management
**Objectif :** Gestion utilisateurs et préférences

**Tasks :**
- [ ] User model (DB schema)
- [ ] Registration/login (simple auth)
- [ ] Profile management UI
- [ ] Topic/preferences CRUD

**Deliverables :**
```python
# User model
class User(Base):
    id: UUID
    email: EmailStr
    topics: List[Topic]
    frequency: Literal["daily", "weekly"]
    delivery_time: time
    filters: Dict  # journals, pub_types, etc.
    created_at: datetime
    
# API
POST /users/register
POST /users/login
GET  /users/me
PUT  /users/me/topics
```

**Critères de Succès :**
- ✅ User CRUD fonctionnel
- ✅ Auth sécurisé (JWT tokens)
- ✅ Preferences sauvegardées
- ✅ 10 test users créés

---

#### **Phase 3 : Interface (Semaines 9-12)**

##### Semaine 9 : Streamlit Dashboard MVP
**Objectif :** UI simple pour interagir avec le système

**Tasks :**
- [ ] Pages: Home, Topics, Digest History, Settings
- [ ] Topic management (add/edit/delete)
- [ ] Digest viewer avec filtering
- [ ] Entity visualization (charts)

**Deliverables :**
```python
# streamlit_app.py

import streamlit as st

# Sidebar
st.sidebar.title("medAssist")
page = st.sidebar.radio("Navigation", ["Home", "Topics", "History"])

if page == "Topics":
    st.title("Your Research Topics")
    
    # Add new topic
    with st.form("add_topic"):
        query = st.text_input("Topic query")
        filters = st.multiselect("Filters", ["RCT", "Meta-Analysis"])
        submit = st.form_submit_button("Add Topic")
    
    # Display topics
    for topic in user.topics:
        st.write(f"**{topic.query}**")
        st.write(f"Last updated: {topic.last_run}")
```

**Critères de Succès :**
- ✅ Dashboard accessible en local
- ✅ 4-5 pages fonctionnelles
- ✅ Responsive (desktop/tablet)
- ✅ UX testée avec 5 users

---

##### Semaine 10 : Visualization & Analytics
**Objectif :** Graphs et charts pour insights

**Tasks :**
- [ ] Entity timeline charts (Plotly)
- [ ] Trend visualization
- [ ] Citation network graph (NetworkX)
- [ ] Thematic clusters viz

**Deliverables :**
```python
import plotly.express as px

# Entity timeline
fig = px.line(
    entity_timeline,
    x="date",
    y="count",
    color="entity",
    title="Entity Mentions Over Time"
)
st.plotly_chart(fig)

# Network graph
import networkx as nx
G = build_citation_network(articles)
st.plotly_chart(plot_network(G))
```

**Critères de Succès :**
- ✅ 5+ types de visualizations
- ✅ Interactive (zoom, filter)
- ✅ Export PNG/SVG
- ✅ Performance <2s rendering

---

##### Semaine 11 : Export & Integration
**Objectif :** Export données vers outils externes

**Tasks :**
- [ ] Export CSV/JSON articles
- [ ] Export BibTeX pour Zotero
- [ ] API endpoints pour intégrations
- [ ] Webhooks (Slack, Discord)

**Deliverables :**
```python
# Export endpoints
GET /export/articles?format=csv
GET /export/digest/{digest_id}?format=pdf
GET /export/bibtex?pmids=123,456,789

# Webhooks
POST /webhooks/slack
{
  "channel": "#research",
  "digest_id": "uuid",
  "trigger": "daily"
}
```

**Critères de Succès :**
- ✅ Exports fonctionnent sans erreurs
- ✅ BibTeX valide (importable Zotero)
- ✅ Slack integration testée
- ✅ API documentée (Swagger)

---

##### Semaine 12 : Polish & Testing
**Objectif :** Refinement UX + tests utilisateurs

**Tasks :**
- [ ] UX improvements (feedback round 1)
- [ ] Bug fixes
- [ ] Performance optimizations
- [ ] User testing avec 10 personnes

**Deliverables :**
- Liste 20+ bugs identifiés et fixés
- UX improvements log
- Performance report (load times)
- User feedback summary

**Critères de Succès :**
- ✅ 0 critical bugs
- ✅ Page load <3s
- ✅ User satisfaction >4/5
- ✅ 10 users actifs quotidiennement

---

#### **Phase 4 : Production (Semaines 13-16)**

##### Semaine 13 : Deployment Infrastructure
**Objectif :** Déploiement cloud production-ready

**Tasks :**
- [ ] Setup cloud (AWS/GCP/Azure ou self-hosted)
- [ ] Kubernetes manifests (si scale)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring (Grafana, Sentry)

**Deliverables :**
```yaml
# k8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medAssist-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: medAssist/api:v1.0.0
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

**Critères de Succès :**
- ✅ Déployé sur cloud avec HA
- ✅ Auto-scaling configuré
- ✅ Monitoring dashboards actifs
- ✅ CI/CD déploie en <10min

---

##### Semaine 14 : Security & Compliance
**Objectif :** Sécurisation et conformité RGPD

**Tasks :**
- [ ] HTTPS/TLS certificates
- [ ] Rate limiting API
- [ ] Data encryption at rest
- [ ] GDPR compliance (export, delete user data)

**Deliverables :**
```python
# GDPR endpoints
GET  /users/me/data  # Export all user data
DELETE /users/me     # Delete account + data

# Security
- SSL/TLS: A+ rating (SSL Labs)
- Rate limit: 100 req/hour per user
- Encryption: AES-256 for sensitive data
```

**Critères de Succès :**
- ✅ Security audit passé
- ✅ GDPR-compliant
- ✅ Penetration testing OK
- ✅ Privacy policy published

---

##### Semaine 15 : Documentation & Onboarding
**Objectif :** Docs complètes pour users et devs

**Tasks :**
- [ ] User guide (Getting Started, FAQ)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Developer docs (architecture, contrib guide)
- [ ] Video tutorials (3-5 min each)

**Deliverables :**
```markdown
# Docs structure
/docs
  /user-guide
    - getting-started.md
    - managing-topics.md
    - understanding-digests.md
    - faq.md
  /api
    - authentication.md
    - endpoints.md
    - webhooks.md
  /developer
    - architecture.md
    - contributing.md
    - local-setup.md
  /videos
    - 01-quick-start.mp4
    - 02-adding-topics.mp4
    - 03-reading-digests.mp4
```

**Critères de Succès :**
- ✅ 15+ doc pages
- ✅ API 100% documentée
- ✅ 3 video tutorials
- ✅ New user peut setup en <15min

---

##### Semaine 16 : Launch & Marketing
**Objectif :** Lancement public et acquisition users

**Tasks :**
- [ ] Landing page (Next.js/marketing site)
- [ ] Blog post technique (Medium/Dev.to)
- [ ] Social media (Twitter, LinkedIn)
- [ ] Product Hunt launch

**Deliverables :**
```markdown
# Launch checklist
- [ ] Landing page live (medAssist.ai)
- [ ] Blog post published (3 platforms)
- [ ] Tweet thread (10+ tweets)
- [ ] Product Hunt submission
- [ ] Reddit post (r/bioinformatics)
- [ ] Hacker News Show HN
- [ ] Email 50 potential users
```

**Critères de Succès :**
- ✅ 100+ signups first week
- ✅ Featured on Product Hunt
- ✅ 500+ upvotes/stars
- ✅ 10+ paying users (if premium tier)

---

## 8. Plan de Validation

### ✅ Stratégie de Validation Multi-Niveaux

#### Niveau 1 : Validation Technique (Automatique)

**Métriques :**
```python
# NER Accuracy
ner_metrics = {
    "precision": 0.92,
    "recall": 0.89,
    "f1_score": 0.90,
    # Baseline: OpenMed benchmarks
}

# Search Relevance
search_metrics = {
    "precision@10": 0.85,  # Top 10 results relevant
    "ndcg": 0.88,          # Normalized Discounted Cumulative Gain
}

# Clustering Quality
cluster_metrics = {
    "silhouette_score": 0.65,  # -1 to 1, higher better
    "davies_bouldin_index": 0.8  # Lower better
}

# System Performance
perf_metrics = {
    "digest_generation_time": "8.5s",
    "api_response_time_p95": "450ms",
    "uptime": "99.5%"
}
```

**Tests Automatisés :**
- Unit tests: 200+ tests
- Integration tests: 50+ scenarios
- E2E tests: 10+ user flows
- Performance tests: Load testing 100 concurrent users

---

#### Niveau 2 : Validation Utilisateur (Qualitative)

**Phase Alpha (Semaines 5-8) - 5 Users**

**Profil testeurs :**
- 2 doctorants biologie/médecine
- 1 chercheur senior
- 1 médecin-chercheur
- 1 data scientist biotech

**Questionnaire :**
```yaml
Questions:
  - "Le digest contient-il des informations pertinentes?" (1-5)
  - "Les articles sont-ils bien catégorisés par thème?" (1-5)
  - "Les tendances détectées sont-elles significatives?" (1-5)
  - "Quelle est la probabilité que vous utilisiez cet outil quotidiennement?" (1-10)
  - "Quelles features manquent le plus?"
  
Métriques Cibles:
  - Pertinence: >4/5
  - Catégorisation: >4/5
  - Tendances: >3.5/5
  - Likelihood to use: >7/10
```

**Feedback Session :**
- 1h interview individuel
- Screen recording usage
- Identify pain points
- Feature requests

---

**Phase Beta (Semaines 9-12) - 20 Users**

**Critères d'Entrée :**
- Chercheurs actifs (>1 publi/an)
- Domaines variés (oncologie, génétique, micro-bio, etc.)
- Mix juniors/seniors

**A/B Testing :**
```python
# Test différentes configurations
groups = {
    "A": {  # Digest court
        "summary_length": 300,
        "articles_count": 10,
        "trends_threshold": 0.7
    },
    "B": {  # Digest long
        "summary_length": 500,
        "articles_count": 20,
        "trends_threshold": 0.5
    }
}

# Mesurer engagement
engagement_metrics = {
    "open_rate": 0.75,      # Email opened
    "click_rate": 0.45,     # Clicked article link
    "read_time": "4.2 min", # Time on digest
    "retention_7d": 0.80    # Still using after 7 days
}
```

**Objectifs Beta :**
- 80%+ satisfaction
- 70%+ retention hebdomadaire
- 50%+ click-through rate
- <5 critical bugs

---

#### Niveau 3 : Validation Scientifique (Gold Standard)

**Benchmark Dataset :**
```python
# Créer test set gold standard
gold_set = {
    "topic": "Cancer immunotherapy",
    "period": "Jan 2025 - Jan 2026",
    "articles": 500,  # Manually curated by experts
    "ground_truth": {
        "themes": [
            "CAR-T therapy",
            "Checkpoint inhibitors",
            "Cancer vaccines",
            "Bispecific antibodies"
        ],
        "key_entities": {
            "GENE": ["PD-1", "PD-L1", "CTLA-4"],
            "DRUG": ["pembrolizumab", "nivolumab"],
            "DISEASE": ["melanoma", "NSCLC"]
        },
        "trending": [
            {"entity": "CAR-macrophages", "expected_growth": "+150%"}
        ]
    }
}

# Evaluation
results = evaluate_system(gold_set)
print(f"Theme detection accuracy: {results.theme_accuracy}")
print(f"Entity extraction F1: {results.entity_f1}")
print(f"Trend detection precision: {results.trend_precision}")
```

**Comparaison avec Baselines :**
```python
# Comparer vs méthodes existantes
baselines = {
    "PubMed_Alerts": {
        "relevance": 0.65,  # Beaucoup de bruit
        "organization": 0.3, # Pas de clustering
        "trends": 0.0        # Pas de détection
    },
    "Manual_Review": {
        "relevance": 0.95,  # Expert curation
        "organization": 0.9,
        "trends": 0.7,
        "time_cost": "10 hours/week"  # But expensive
    },
    "medAssist": {
        "relevance": 0.85,  # Target
        "organization": 0.80,
        "trends": 0.75,
        "time_cost": "15 min/week"  # Big win
    }
}

# medAssist target: 85-90% quality of manual at 2% time cost
```

---

## 9. Stratégie de Déploiement

### 🚀 Options de Déploiement

#### Option 1 : Self-Hosted (Open-Source)
**Pour qui :** Chercheurs individuels, petits labs

**Stack :**
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    image: medAssist/api:latest
    ports: ["8000:8000"]
    environment:
      - PUBMED_EMAIL=user@email.com
      - ANTHROPIC_API_KEY=sk-xxx
  
  postgres:
    image: postgres:15
    volumes: ["./data/postgres:/var/lib/postgresql/data"]
  
  redis:
    image: redis:7
  
  chroma:
    image: chromadb/chroma:latest
    volumes: ["./data/chroma:/chroma/chroma"]
  
  celery-worker:
    image: medAssist/api:latest
    command: celery -A app.celery worker
  
  celery-beat:
    image: medAssist/api:latest
    command: celery -A app.celery beat
```

**Installation :**
```bash
# Clone repo
git clone https://github.com/username/medAssist.git
cd medAssist

# Configure
cp .env.example .env
# Edit .env with your PUBMED_EMAIL, ANTHROPIC_API_KEY

# Run
docker-compose up -d

# Access at http://localhost:8000
```

**Coût :** Free (except Anthropic API ~10$/month)

---

#### Option 2 : Managed Hosting (SaaS)
**Pour qui :** Users non-techniques, équipes

**Stack :**
- Frontend: Vercel
- Backend: Railway / Render
- Database: Supabase (Postgres) / PlanetScale
- Vector Store: Pinecone
- Queue: Upstash Redis
- Storage: Cloudflare R2

**Tiers :**
```yaml
Free Tier:
  - 2 topics
  - Daily digest only
  - 50 articles/day
  - Email delivery
  - Price: $0/month

Pro Tier:
  - 10 topics
  - Daily + Weekly + Custom
  - 500 articles/day
  - Email + Dashboard + API
  - Export features
  - Price: $15/month

Team Tier:
  - Unlimited topics
  - Shared workspaces
  - 5000 articles/day
  - Priority support
  - Custom integrations
  - Price: $49/month (5 users)
```

---

#### Option 3 : Hybrid (Best of Both)
**Pour qui :** Institutions, universités

**Model :**
- Core engine: Open-source self-hosted
- Premium features: SaaS addon
- Data: Stays on-premise (compliance)

**Example :**
```
University of Paris-Saclay deployment:
- Self-hosted core (GDPR compliance)
- 500 researchers
- SaaS addon: Advanced analytics dashboard
- Cost: $500/month (enterprise tier)
```

---

### 📊 Scaling Strategy

#### Phase 1 : MVP (0-100 users)
**Infrastructure :**
- Single VPS (4 CPU, 16GB RAM)
- Docker Compose
- Cost: ~50€/month

**Capacity :**
- 100 daily digests
- 10,000 articles indexed
- 1,000 API calls/day

---

#### Phase 2 : Growth (100-1000 users)
**Infrastructure :**
- Kubernetes cluster (3 nodes)
- Managed PostgreSQL
- Pinecone vector store
- Load balancer

**Cost : ~300€/month**

**Capacity :**
- 1,000 daily digests
- 100,000 articles indexed
- 10,000 API calls/day

---

#### Phase 3 : Scale (1000+ users)
**Infrastructure :**
- Multi-region Kubernetes
- Read replicas (PostgreSQL)
- CDN (Cloudflare)
- Advanced monitoring

**Cost : ~1000€/month**

**Capacity :**
- 10,000+ daily digests
- 1M+ articles indexed
- 100,000 API calls/day

---

## 10. Monétisation & Business Model

### 💰 Revenue Streams

#### 1. Freemium SaaS (Primary)
```yaml
Free:
  - 2 topics
  - Daily digest
  - Basic features
  - Revenue: $0
  - Target: 1000 users
  - Conversion goal: 10% to Pro

Pro ($15/month):
  - 10 topics
  - All digest frequencies
  - Advanced filters
  - Export features
  - Revenue: $15 × 100 users = $1,500/month

Team ($49/month):
  - Unlimited topics
  - Shared workspaces
  - API access
  - Priority support
  - Revenue: $49 × 20 teams = $980/month

Total MRR (Year 1): ~$2,500/month
```

---

#### 2. API Access (Secondary)
```yaml
Pricing:
  - $0.01 per digest generated
  - $0.001 per article analyzed
  - $0.10 per 1000 API calls

Use Case: 
  - Biotech companies integrating into internal tools
  - Research platforms adding medAssist as feature
  
Revenue Potential: $500-2000/month (Year 1)
```

---

#### 3. Enterprise Licenses (Long-term)
```yaml
Target: Universities, pharma, biotech

Offering:
  - On-premise deployment
  - Custom features
  - SLA guarantees
  - Dedicated support
  - Training sessions

Pricing: $5,000-20,000/year per institution

Revenue Potential: $50,000+/year (Year 2-3)
```

---

#### 4. Consulting & Custom Development
```yaml
Services:
  - Custom intelligence agents
  - Integration with internal tools
  - Training workshops
  - White-label solutions

Pricing: $150-300/hour

Revenue Potential: $10,000-50,000/year (opportunistic)
```

---

### 📈 Financial Projections (3 Years)

#### Year 1 : MVP → Product-Market Fit
```yaml
Users:
  Free: 500
  Pro: 50 ($750/month)
  Team: 5 ($245/month)

MRR: $1,000
ARR: $12,000

Costs:
  - Infrastructure: $3,600/year
  - APIs (Anthropic): $2,400/year
  - Marketing: $1,000/year
  Total: $7,000/year

Net: $5,000/year (break-even+)
```

---

#### Year 2 : Growth
```yaml
Users:
  Free: 2,000
  Pro: 200 ($3,000/month)
  Team: 20 ($980/month)
  Enterprise: 2 ($20,000/year)

MRR: $4,000
ARR: $48,000 + $40,000 (enterprise) = $88,000

Costs:
  - Infrastructure: $12,000/year
  - APIs: $10,000/year
  - Marketing: $5,000/year
  - Part-time dev: $20,000/year
  Total: $47,000/year

Net: $41,000/year
```

---

#### Year 3 : Scale
```yaml
Users:
  Free: 5,000
  Pro: 500 ($7,500/month)
  Team: 50 ($2,450/month)
  Enterprise: 10 ($100,000/year)

MRR: $10,000
ARR: $120,000 + $100,000 = $220,000

Costs:
  - Infrastructure: $30,000/year
  - APIs: $25,000/year
  - Marketing: $15,000/year
  - Team (2 devs): $80,000/year
  Total: $150,000/year

Net: $70,000/year
Profitable sustainable business 🎉
```

---

## 11. Risques & Mitigations

### ⚠️ Risques Identifiés

#### 1. Dépendance API PubMed
**Risque :** PubMed rate limits, downtime, policy changes

**Impact :** CRITICAL (système inutilisable)

**Probabilité :** MEDIUM

**Mitigation :**
- Obtenir NCBI API key officielle (10 req/s vs 3)
- Aggressive caching (Redis, 24-48h)
- Fallback vers Europe PMC API
- Politeness policy (respect rate limits)

---

#### 2. Coût LLM API (Anthropic)
**Risque :** Coûts explosent avec croissance users

**Impact :** HIGH (rentabilité menacée)

**Probabilité :** MEDIUM

**Mitigation :**
- Caching résumés similaires
- Batch processing (reduce API calls)
- Tier limits (Free = 1 digest/day, Pro = unlimited)
- Fallback vers modèle open-source (Llama, Mistral) si budget serré

**Calcul coûts :**
```python
# Anthropic Claude Sonnet 4 pricing
input_price = 3.00  # per 1M tokens
output_price = 15.00  # per 1M tokens

# Digest moyen
input_tokens = 10000  # articles + metadata
output_tokens = 1000  # summary

cost_per_digest = (
    (input_tokens / 1_000_000) * input_price +
    (output_tokens / 1_000_000) * output_price
)  # = $0.045 per digest

# 1000 users × 30 digests/month = 30,000 digests
monthly_cost = 30000 * 0.045  # = $1,350/month

# Gérable si $15/month Pro subscription × 200 users = $3,000 MRR
```

---

#### 3. Qualité NER (OpenMed Errors)
**Risque :** Entités mal extraites → résumés faux

**Impact :** MEDIUM (satisfaction users)

**Probabilité :** LOW (OpenMed SOTA)

**Mitigation :**
- Confidence threshold élevé (0.7+)
- Human-in-the-loop spot checks
- User feedback "Report error"
- A/B test différents modèles NER

---

#### 4. Compétition
**Risque :** Gros acteurs (Elsevier, Springer) lancent service similaire

**Impact :** HIGH

**Probabilité :** MEDIUM (long-term)

**Mitigation :**
- First-mover advantage (build community)
- Open-source = hard to kill (GitHub stars, contributors)
- Focus niche (chercheurs, pas grand public)
- Integration ecosystem (Zotero, Mendeley)

---

#### 5. GDPR / Data Privacy
**Risque :** Non-compliance → amendes, shutdown

**Impact :** CRITICAL

**Probabilité :** LOW (if designed properly)

**Mitigation :**
- Privacy by design
- Minimal data collection
- GDPR-compliant from day 1
- User data export/delete endpoints
- EU hosting option

---

## 12. Extensions Futures

### 🔮 Roadmap Long-Terme (12-24 mois)

#### Extension 1 : Multi-Source Intelligence
**Description :** Extend beyond PubMed

**Sources :**
- bioRxiv / medRxiv (preprints)
- ClinicalTrials.gov (essais cliniques)
- FDA approvals / EMA decisions
- Patents (Google Patents, Lens.org)
- Conferences (abstracts)
- Twitter/X (scientific discourse)

**Valeur :** Vue 360° du landscape

---

#### Extension 2 : Collaborative Workspaces
**Description :** Team features

**Features :**
- Shared topics/digests
- Assign articles to team members
- Comments/annotations
- Task management ("Marie, review this article")
- Slack/Teams integration

**Use Case :** Lab de 10 chercheurs collabore sur projet

---

#### Extension 3 : Predictive Analytics
**Description :** ML pour prédire futures tendances

**Features :**
- "X will be trending in 3 months" (forecast)
- Identify researchers before they're famous
- Predict FDA approval likelihood (based on trial results)

**Technique :**
- Time series forecasting (Prophet, ARIMA)
- Graph neural networks (citation networks)

---

#### Extension 4 : Clinical Decision Support
**Description :** Évolution vers use case clinique

**Features :**
- Patient case → literature search
- Treatment recommendations with evidence
- Drug interaction checking
- Guideline compliance

**Requirement :**
- Medical expert validation
- HIPAA compliance
- FDA/EMA approval potential

---

#### Extension 5 : Knowledge Graph
**Description :** Build biomedical knowledge graph

**Structure :**
```
(Disease) -[TREATED_BY]-> (Drug)
(Drug) -[TARGETS]-> (Gene)
(Gene) -[MUTATED_IN]-> (Disease)
(Researcher) -[PUBLISHED]-> (Article)
(Article) -[CITES]-> (Article)
```

**Use Cases :**
- "Show me all drugs targeting EGFR"
- "Find alternative treatments for X disease"
- "Who are the key researchers in Y field?"

**Tech :** Neo4j, GraphQL API

---

#### Extension 6 : Marketplace
**Description :** Ecosystem of intelligence agents

**Concept :**
- Users create custom agents
- Share/sell agents on marketplace
- Example agents:
  - "Clinical Trial Tracker"
  - "Competitor Monitor (Pharma)"
  - "Grant Opportunity Finder"
  - "Literature Review Generator"

**Revenue :** 30% commission on paid agents

---

## 📞 Contact & Collaboration

**Projet :** medAssist - Intelligent Biomedical Intelligence  
**Status :** In Development (Phase 1)  
**License :** Apache 2.0 (Open Source)  
**Repository :** github.com/username/medAssist  

**Cherche :**
- Beta testers (chercheurs, doctorants)
- Contributors (Python, ML, bioinformatics)
- Advisors (médecins, chercheurs senior)
- Funding (grants, investors)

---

## 📚 Ressources & Références

### Papers & Research
- OpenMed NER: arXiv:2508.01630
- PubMed E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- Biomedical NER Survey: https://academic.oup.com/bib/article/21/6/1954/5645825

### Outils & APIs
- OpenMed: https://openmed.life
- Biopython: https://biopython.org
- Anthropic Claude: https://anthropic.com
- ChromaDB: https://www.trychroma.com

### Inspiration
- F1000 Recommendations
- PubMed Clinical Queries
- Semantic Scholar
- Connected Papers

---

**Fin du Document - Version 1.0**  
**Dernière mise à jour : Février 2026**
