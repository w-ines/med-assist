#!/usr/bin/env python3
"""
Test script for Deep Agents integration - Updated version.
Tests the complete flow: router → agent → tool → RAG
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: Verify all imports work"""
    print("=" * 60)
    print("TEST 1: Vérification des imports")
    print("=" * 60)
    
    try:
        print("✓ Importing deepagents router...")
        from deepagents.router import router as deepagent_router
        print("  ✅ Router imported successfully")
        
        print("✓ Importing main_agent...")
        from deepagents.agents.main_agent import create_medAssist_agent
        print("  ✅ Main agent imported successfully")
        
        print("✓ Importing rag_tool...")
        from deepagents.tools.knowledge.rag_tool import retrieve_knowledge
        print("  ✅ RAG tool imported successfully")
        
        print("✓ Importing rag module...")
        from rag.retriever import get_retriever
        from rag.vector_store import get_vector_store
        print("  ✅ RAG module imported successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_tool():
    """Test 2: Test RAG tool directly"""
    print("\n" + "=" * 60)
    print("TEST 2: Test du tool RAG directement")
    print("=" * 60)
    
    try:
        from deepagents.tools.knowledge.rag_tool import retrieve_knowledge
        
        print("✓ Calling retrieve_knowledge with test query...")
        result = retrieve_knowledge.invoke({
            "query": "What is RAG?",
            "top_k": 3,
            "enable_kg_enrichment": False  # Disable KG for simple test
        })
        
        print(f"  ✅ Tool executed successfully")
        print(f"  📊 Results: {len(result.get('results', []))} documents retrieved")
        print(f"  📝 Context length: {len(result.get('context', ''))} chars")
        
        if result.get('error'):
            print(f"  ⚠️  Warning: {result['error']}")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test 3: Test agent creation"""
    print("\n" + "=" * 60)
    print("TEST 3: Test de création de l'agent")
    print("=" * 60)
    
    try:
        print("✓ Creating medAssist agent...")
        from deepagents.agents.main_agent import create_medAssist_agent
        
        agent = create_medAssist_agent()
        print("  ✅ Agent created successfully")
        print(f"  🤖 Agent type: {type(agent).__name__}")
        
        # Check if it's an AgentExecutor
        from langchain.agents import AgentExecutor
        if isinstance(agent, AgentExecutor):
            print("  ✅ Agent is a valid LangChain AgentExecutor")
        
        return True
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_invoke():
    """Test 4: Test agent invocation"""
    print("\n" + "=" * 60)
    print("TEST 4: Test d'invocation de l'agent")
    print("=" * 60)
    
    try:
        from deepagents.agents.main_agent import create_medAssist_agent
        
        print("✓ Creating agent...")
        agent = create_medAssist_agent()
        
        print("✓ Invoking agent with simple query...")
        result = agent.invoke({"input": "Hello, what can you do?"})
        
        print("  ✅ Agent invoked successfully")
        
        # Extract response
        if isinstance(result, dict):
            response = result.get("output", str(result))
            print(f"  💬 Response preview: {response[:200]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Agent invocation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_health():
    """Test 5: Test router health endpoint"""
    print("\n" + "=" * 60)
    print("TEST 5: Test du health check du router")
    print("=" * 60)
    
    try:
        from deepagents.router import router
        
        print("✓ Router imported successfully")
        print(f"  📍 Router prefix: {router.prefix if hasattr(router, 'prefix') else 'None'}")
        print(f"  🛣️  Routes: {len(router.routes)} routes registered")
        
        # List routes
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                print(f"    - {list(route.methods)[0] if route.methods else 'GET'} {route.path}")
        
        return True
    except Exception as e:
        print(f"  ❌ Router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪" * 30)
    print("DEEP AGENTS INTEGRATION TEST SUITE")
    print("🧪" * 30 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("RAG Tool", test_rag_tool),
        ("Agent Creation", test_agent_creation),
        ("Agent Invocation", test_agent_invoke),
        ("Router Health", test_router_health),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés ! L'intégration Deep Agents est fonctionnelle.")
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
