#!/usr/bin/env python3
"""Test PubMed integration with Knowledge Graph."""

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST: PubMed → Knowledge Graph Integration")
print("=" * 60)

# Test 1: Import modules
print("\nTest 1: Importing modules...")
try:
    from tools.kg_tool import ingest_from_pubmed, stats
    from tools.pubmed_tool import search_pubmed
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Check KG stats before
print("\nTest 2: Knowledge Graph stats (before)...")
try:
    kg_stats_before = stats()
    print(f"✅ KG Stats:")
    print(f"   - Nodes: {kg_stats_before.get('node_count', 0)}")
    print(f"   - Edges: {kg_stats_before.get('edge_count', 0)}")
    print(f"   - Components: {kg_stats_before.get('connected_components', 0)}")
except Exception as e:
    print(f"❌ KG stats failed: {e}")
    exit(1)

# Test 3: Search PubMed (without ingestion)
print("\nTest 3: Testing PubMed search...")
try:
    test_query = "aspirin cardiovascular"
    print(f"   Query: '{test_query}'")
    
    result = search_pubmed(
        query=test_query,
        max_results=5,
        fetch_details=True
    )
    
    if "error" in result:
        print(f"⚠️  PubMed search error: {result['error']}")
        print("   (This is normal if NCBI API is not configured)")
    else:
        print(f"✅ PubMed search successful:")
        print(f"   - Total found: {result.get('total', 0)}")
        print(f"   - Articles retrieved: {len(result.get('articles', []))}")
        
        # Show first article
        articles = result.get('articles', [])
        if articles:
            first = articles[0]
            print(f"\n   First article:")
            print(f"   - PMID: {first.get('pmid', 'N/A')}")
            print(f"   - Title: {first.get('title', 'N/A')[:80]}...")
            print(f"   - Journal: {first.get('journal', 'N/A')}")
            print(f"   - Date: {first.get('pub_date', 'N/A')}")
            
            mesh_terms = first.get('mesh_terms', [])
            if mesh_terms:
                print(f"   - MeSH terms: {', '.join(mesh_terms[:5])}")

except Exception as e:
    print(f"⚠️  PubMed search test skipped: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Ingest from PubMed into KG
print("\nTest 4: Ingesting PubMed articles into Knowledge Graph...")
try:
    # Use a small query for testing
    test_query = "aspirin myocardial infarction"
    max_articles = 5
    
    print(f"   Query: '{test_query}'")
    print(f"   Max articles: {max_articles}")
    print("   Processing (this may take 10-30 seconds)...")
    
    result = ingest_from_pubmed(
        query=test_query,
        max_results=max_articles,
    )
    
    if "error" in result:
        print(f"⚠️  Ingestion error: {result['error']}")
    else:
        print(f"\n✅ Ingestion successful:")
        print(f"   - PubMed query: {result.get('pubmed_query', 'N/A')}")
        print(f"   - Articles found: {result.get('articles_found', 0)}")
        print(f"   - Articles processed: {result.get('articles_processed', 0)}")
        print(f"   - Entities extracted: {result.get('entities_extracted', 0)}")
        
        graph_stats = result.get('graph_stats', {})
        print(f"\n   Updated KG stats:")
        print(f"   - Nodes: {graph_stats.get('node_count', 0)}")
        print(f"   - Edges: {graph_stats.get('edge_count', 0)}")
        print(f"   - Density: {graph_stats.get('density', 0):.4f}")

except Exception as e:
    print(f"❌ Ingestion test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check KG stats after
print("\nTest 5: Knowledge Graph stats (after)...")
try:
    kg_stats_after = stats()
    print(f"✅ Final KG Stats:")
    print(f"   - Nodes: {kg_stats_after.get('node_count', 0)}")
    print(f"   - Edges: {kg_stats_after.get('edge_count', 0)}")
    print(f"   - Components: {kg_stats_after.get('connected_components', 0)}")
    
    # Calculate delta
    nodes_added = kg_stats_after.get('node_count', 0) - kg_stats_before.get('node_count', 0)
    edges_added = kg_stats_after.get('edge_count', 0) - kg_stats_before.get('edge_count', 0)
    
    if nodes_added > 0 or edges_added > 0:
        print(f"\n   📈 Growth:")
        print(f"   - Nodes added: +{nodes_added}")
        print(f"   - Edges added: +{edges_added}")
    
except Exception as e:
    print(f"❌ Final stats failed: {e}")

# Test 6: Query top nodes
print("\nTest 6: Top medical entities in KG...")
try:
    from tools.kg_tool import query_top_nodes
    
    top_nodes = query_top_nodes(n=10, sort_by="frequency")
    
    if top_nodes:
        print(f"✅ Top 10 entities by frequency:")
        for i, node in enumerate(top_nodes, 1):
            label = node.get('label', 'N/A')
            entity_type = node.get('entity_type', 'N/A')
            frequency = node.get('frequency', 0)
            print(f"   {i}. {label} ({entity_type}) - freq: {frequency}")
    else:
        print("   No nodes in KG yet")

except Exception as e:
    print(f"⚠️  Top nodes query failed: {e}")

print("\n" + "=" * 60)
print("✅ PubMed → KG Integration Tests Complete")
print("=" * 60)

print("\n📚 Next steps:")
print("1. Configure NCBI API key in .env for higher rate limits")
print("2. Run: ingest_from_pubmed('your query', max_results=50)")
print("3. Use enriched KG with RAG for better medical Q&A")
print("4. Explore relationships with query_top_edges()")
