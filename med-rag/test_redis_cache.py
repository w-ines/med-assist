#!/usr/bin/env python3
"""Test Redis cache for Knowledge Graph."""

import time
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST: Cache Redis pour Knowledge Graph")
print("=" * 60)

# Test 1: Vérifier la connexion Redis
print("\nTest 1: Connexion Redis")
from storage.kg_cache_redis import get_cache_stats

stats = get_cache_stats()
if stats.get("connected"):
    print(f"✅ Redis connecté")
    print(f"   Hit rate: {stats.get('hit_rate', 0):.2f}%")
else:
    print(f"❌ Redis non connecté: {stats.get('error', 'Unknown')}")
    print("   Assure-toi que REDIS_URL est configuré dans .env")
    print("   Exemple: REDIS_URL=redis://localhost:6379/0")
    exit(1)

# Test 2: Premier chargement (cache miss)
print("\nTest 2: Premier chargement (cache miss)")
from tools.kg_tool import stats as kg_stats

start = time.time()
result1 = kg_stats()
duration1 = time.time() - start

print(f"✅ Stats: {result1}")
print(f"   Durée: {duration1*1000:.2f}ms")

# Test 3: Deuxième chargement (cache hit)
print("\nTest 3: Deuxième chargement (cache hit)")

start = time.time()
result2 = kg_stats()
duration2 = time.time() - start

print(f"✅ Stats: {result2}")
print(f"   Durée: {duration2*1000:.2f}ms")

# Calcul du speedup
if duration1 > 0:
    speedup = duration1 / duration2
    print(f"\n🚀 Speedup: {speedup:.2f}x plus rapide avec cache")
    
    if speedup > 2:
        print("   ✅ Cache Redis fonctionne parfaitement!")
    elif speedup > 1.2:
        print("   ⚠️  Cache fonctionne mais amélioration modérée")
    else:
        print("   ❌ Cache ne semble pas accélérer les requêtes")

# Test 4: Ingest et invalidation du cache
print("\nTest 4: Ingest et invalidation du cache")
from tools.kg_tool import ingest_text

result = ingest_text(
    'Metformin treats diabetes',
    source='TEST_CACHE',
    provider='gliner',
    entity_types=['DRUG', 'DISEASE']
)

print(f"✅ Ingest: {result}")

# Test 5: Rechargement après invalidation (cache miss)
print("\nTest 5: Rechargement après invalidation (cache miss)")

start = time.time()
result3 = kg_stats()
duration3 = time.time() - start

print(f"✅ Stats: {result3}")
print(f"   Durée: {duration3*1000:.2f}ms")

if duration3 > duration2 * 1.5:
    print("   ✅ Cache invalidé correctement après ingest")
else:
    print("   ⚠️  Cache peut ne pas avoir été invalidé")

# Test 6: Stats finales du cache
print("\nTest 6: Stats finales du cache Redis")
final_stats = get_cache_stats()

print(f"✅ Statistiques Redis:")
print(f"   Hits: {final_stats.get('keyspace_hits', 0)}")
print(f"   Misses: {final_stats.get('keyspace_misses', 0)}")
print(f"   Hit rate: {final_stats.get('hit_rate', 0):.2f}%")
print(f"   Total commands: {final_stats.get('total_commands_processed', 0)}")

print("\n" + "=" * 60)
print("✅ Tests terminés")
print("=" * 60)
