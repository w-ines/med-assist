"""
Main medAssist Deep Agent - Biomedical Intelligence Agent.
Replaces smolagents CodeAgent with LangChain Deep Agents.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


def create_medAssist_agent(
    tools: Optional[List] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_steps: int = 5,
):
    """
    Create the main medAssist Deep Agent with biomedical tools.
    
    Args:
        tools: List of LangChain tools (if None, uses default tools)
        model_name: LLM model name (defaults to env OPEN_AI_MODEL)
        temperature: LLM temperature (default: 0.3 for focused responses)
        max_steps: Maximum agent steps (default: 5)
        
    Returns:
        AgentExecutor (LangChain agent)
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.runnables import RunnablePassthrough
    
    # Import available tools
    from deepagents.tools.knowledge.rag_tool import retrieve_knowledge
    
    # Configure LLM
    model_name = model_name or os.getenv("OPEN_AI_MODEL", "gpt-4o")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPEN_ROUTER_KEY or OPENAI_API_KEY environment variable."
        )
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )
    
    # Default tools if none provided
    if tools is None:
        tools = [
            retrieve_knowledge,
            # More tools will be added as they are migrated
        ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a simple agent executor wrapper with conversation memory
    class SimpleAgentExecutor:
        def __init__(self, llm_with_tools, tools, max_iterations=5, conversation_id=None, stream_callback=None):
            self.llm = llm_with_tools
            self.tools = {tool.name: tool for tool in tools}
            self.max_iterations = max_iterations
            self.conversation_id = conversation_id or "default"
            self.stream_callback = stream_callback  # Callback pour streamer les étapes
            
            # Conversation memory (in-memory for now)
            # TODO: Migrate to Supabase for persistence
            if not hasattr(SimpleAgentExecutor, '_conversations'):
                SimpleAgentExecutor._conversations = {}
            
            if self.conversation_id not in SimpleAgentExecutor._conversations:
                SimpleAgentExecutor._conversations[self.conversation_id] = []
            
            self.system_prompt = """You are medAssist, an intelligent biomedical research assistant.

Your capabilities:
- Search and retrieve information from uploaded medical documents (use retrieve_knowledge)
- Analyze medical literature and research papers
- Extract and understand medical entities (diseases, drugs, genes, proteins)
- Provide evidence-based answers with proper citations

Guidelines:
1. **Always cite sources**: Use [Source N] format when referencing documents
2. **Be precise**: Medical information requires accuracy - cite evidence
3. **Use Knowledge Graph**: When available, leverage entity relationships for deeper insights
4. **Admit uncertainty**: If information is not in the documents, say so clearly
5. **Efficient tool use**: Call retrieve_knowledge with appropriate parameters

Workflow for medical queries:
1. Understand the question and identify key medical entities
2. Use retrieve_knowledge to search documents (with KG enrichment enabled)
3. Analyze retrieved context and relationships
4. Synthesize findings with proper citations
5. Provide actionable insights

Remember: You have access to a Knowledge Graph that connects diseases, drugs, genes, and other medical entities. Use this to provide richer, more connected answers."""
        
        def invoke(self, input_dict):
            """Execute the agent with tool calling loop."""
            user_input = input_dict.get("input", "")
            
            # Build messages with conversation history
            messages = [HumanMessage(content=self.system_prompt)]
            
            # Ensure conversation_id exists in _conversations
            if self.conversation_id not in SimpleAgentExecutor._conversations:
                SimpleAgentExecutor._conversations[self.conversation_id] = []
            
            # Add conversation history (last 10 messages to avoid context overflow)
            history = SimpleAgentExecutor._conversations[self.conversation_id]
            if history:
                messages.extend(history[-10:])  # Keep last 10 messages
            
            # Add current user message
            messages.append(HumanMessage(content=user_input))
            
            for iteration in range(self.max_iterations):
                # Stream thought
                if self.stream_callback:
                    self.stream_callback({"type": "thought", "content": f" Thinking... (step {iteration + 1}/{self.max_iterations})"})
                
                # Call LLM
                response = self.llm.invoke(messages)
                messages.append(response)
                
                # Check if LLM wants to use tools
                if not response.tool_calls:
                    # No tool calls, return final answer
                    if self.stream_callback:
                        self.stream_callback({"type": "answer", "content": response.content})
                    
                    # Save conversation to history
                    SimpleAgentExecutor._conversations[self.conversation_id].append(
                        HumanMessage(content=user_input)
                    )
                    SimpleAgentExecutor._conversations[self.conversation_id].append(
                        AIMessage(content=response.content)
                    )
                    return {"output": response.content}
                
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Stream action
                    if self.stream_callback:
                        args_preview = str(tool_args)[:100]
                        self.stream_callback({"type": "action", "content": f" Using tool: {tool_name}", "args": args_preview})
                    
                    if tool_name in self.tools:
                        try:
                            # Execute tool
                            tool_result = self.tools[tool_name].invoke(tool_args)
                            
                            # Stream observation with detailed preview
                            if self.stream_callback:
                                result_str = str(tool_result)
                                # Get first 300 chars for preview
                                result_preview = result_str[:300] + "..." if len(result_str) > 300 else result_str
                                # Count results if it's a list/array
                                result_count = ""
                                if isinstance(tool_result, (list, tuple)):
                                    result_count = f" ({len(tool_result)} items)"
                                elif isinstance(tool_result, dict) and "documents" in tool_result:
                                    result_count = f" ({len(tool_result.get('documents', []))} documents)"
                                
                                self.stream_callback({
                                    "type": "observation", 
                                    "content": f" Tool result received{result_count}",
                                    "preview": result_preview
                                })
                            
                            messages.append(
                                ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call["id"]
                                )
                            )
                        except Exception as e:
                            if self.stream_callback:
                                self.stream_callback({"type": "error", "content": f" Tool error: {str(e)}"})
                            
                            messages.append(
                                ToolMessage(
                                    content=f"Error executing tool: {str(e)}",
                                    tool_call_id=tool_call["id"]
                                )
                            )
            
            # Max iterations reached
            final_response = "I apologize, but I reached the maximum number of steps. Please try rephrasing your question."
            
            # Save conversation to history even if max iterations reached
            SimpleAgentExecutor._conversations[self.conversation_id].append(
                HumanMessage(content=user_input)
            )
            SimpleAgentExecutor._conversations[self.conversation_id].append(
                AIMessage(content=final_response)
            )
            
            return {"output": final_response}
    
    return SimpleAgentExecutor(llm_with_tools, tools, max_iterations=max_steps)


def create_simple_rag_agent(
    model_name: Optional[str] = None,
    temperature: float = 0.3,
):
    """
    Create a simple RAG-only agent (no planning, just retrieval + answer).
    Useful for quick queries that don't need complex reasoning.
    
    Args:
        model_name: LLM model name
        temperature: LLM temperature
        
    Returns:
        Simple agent for RAG queries
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from deepagents.tools.knowledge.rag_tool import retrieve_knowledge
    
    # Configure LLM
    model_name = model_name or os.getenv("OPEN_AI_MODEL", "gpt-4o")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )
    
    # Simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful medical assistant. Use retrieve_knowledge to search documents and answer questions.
Always cite sources using [Source N] format."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    tools = [retrieve_knowledge]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor
