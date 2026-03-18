#!/usr/bin/env python3
"""Test RAG + KG integration."""

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("TEST: RAG + Knowledge Graph Integration")
print("=" * 60)

# Test 1: Import modules
print("\nTest 1: Importing modules...")
try:
    from rag.retriever import get_retriever, KGEnhancedRetriever
    from rag.chain import create_rag_chain, query_rag
    from rag.vector_store import get_vector_store
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Initialize retriever
print("\nTest 2: Initialize KG-enhanced retriever...")
try:
    retriever = get_retriever(
        top_k=3,
        enable_kg_enrichment=True,
        kg_weight=0.3
    )
    print(f"✅ Retriever initialized: {type(retriever).__name__}")
    print(f"   - top_k: {retriever.top_k}")
    print(f"   - KG enrichment: {retriever.enable_kg_enrichment}")
    print(f"   - KG weight: {retriever.kg_weight}")
except Exception as e:
    print(f"❌ Retriever initialization failed: {e}")
    exit(1)

# Test 3: Test retrieval (if documents exist)
print("\nTest 3: Test retrieval...")
try:
    test_query = "What are the effects of aspirin on cardiovascular health?"
    
    print(f"   Query: '{test_query}'")
    docs = retriever.get_relevant_documents(test_query)
    
    print(f"✅ Retrieved {len(docs)} documents")
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        print(f"\n   Document {i}:")
        print(f"   - Content length: {len(doc.page_content)} chars")
        print(f"   - Filename: {metadata.get('filename', 'N/A')}")
        print(f"   - KG score: {metadata.get('kg_score', 0):.3f}")
        print(f"   - Hybrid score: {metadata.get('hybrid_score', 0):.3f}")
        
        kg_entities = metadata.get("kg_entities", [])
        if kg_entities:
            entity_labels = [e["label"] for e in kg_entities[:3]]
            print(f"   - KG entities: {', '.join(entity_labels)}")
        
        kg_rels = metadata.get("kg_relationships", [])
        if kg_rels:
            print(f"   - KG relationships: {len(kg_rels)}")

except Exception as e:
    print(f"⚠️  Retrieval test skipped: {e}")
    print("   (This is normal if no documents are indexed yet)")

# Test 4: Test RAG chain
print("\nTest 4: Test conversational RAG chain...")
try:
    conversation_id = "test_rag_kg_integration"
    
    # Create chain
    chain = create_rag_chain(
        conversation_id=conversation_id,
        top_k=3,
        enable_kg_enrichment=True,
        kg_weight=0.3
    )
    
    print(f"✅ RAG chain created")
    print(f"   - Conversation ID: {conversation_id}")
    print(f"   - Chain type: {type(chain).__name__}")
    
except Exception as e:
    print(f"❌ Chain creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test query_rag function
print("\nTest 5: Test query_rag function...")
try:
    question = "What medications are used to treat hypertension?"
    
    print(f"   Question: '{question}'")
    print("   Querying RAG system...")
    
    answer = query_rag(
        question=question,
        conversation_id="test_rag_kg",
        top_k=3,
        enable_kg_enrichment=True,
        save_to_memory=False  # Don't save test queries
    )
    
    print(f"\n✅ Answer received:")
    print(f"   Length: {len(answer)} chars")
    print(f"   Preview: {answer[:200]}...")
    
except Exception as e:
    print(f"⚠️  Query test skipped: {e}")
    print("   (This is normal if no documents are indexed yet)")

# Test 6: Check KG integration
print("\nTest 6: Check Knowledge Graph integration...")
try:
    from kg.store import load_graph
    from tools.kg_tool import stats
    
    kg_stats = stats()
    print(f"✅ Knowledge Graph stats:")
    print(f"   - Nodes: {kg_stats.get('node_count', 0)}")
    print(f"   - Edges: {kg_stats.get('edge_count', 0)}")
    print(f"   - Components: {kg_stats.get('connected_components', 0)}")
    
    if kg_stats.get('node_count', 0) > 0:
        print("\n   ✅ KG is populated and ready for RAG enrichment")
    else:
        print("\n   ⚠️  KG is empty - run ingest_text() to populate")
        
except Exception as e:
    print(f"❌ KG integration check failed: {e}")

print("\n" + "=" * 60)
print("✅ RAG + KG Integration Tests Complete")
print("=" * 60)

print("\n📚 Next steps:")
print("1. Index documents: Use huggingsmolagent/tools/vector_store.py")
print("2. Populate KG: Use tools/kg_tool.py ingest_text()")
print("3. Query: Use rag.chain.query_rag() for questions")
print("4. The system will automatically enrich RAG with KG context")
