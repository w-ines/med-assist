# huggingsmolagent/tools/search/generate_query.py

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

async def generate_query(messages: List[Dict[str, Any]], llm_model=None) -> str:
    """
    Generate a search query from conversation messages.
    
    Args:
        messages: List of conversation messages
        llm_model: Optional LLM model to generate the query
        
    Returns:
        Generated search query
    """
    # If no messages, return empty string
    if not messages:
        return ""
    
    # Extract the last message (assumed to be the user's query)
    last_message = messages[-1]
    
    if isinstance(last_message, dict) and "content" in last_message:
        content = last_message["content"]
    elif isinstance(last_message, str):
        content = last_message
    else:
        content = str(last_message)
    
    # Simple option: enhance the query with temporal context if needed
    if not llm_model:
        enhanced_query = _enhance_query_with_context(content)
        return enhanced_query
    
    # Advanced option: use an LLM to generate a better query
    try:
        prompt = f"""
        Based on the conversation history, generate a search query that would help find relevant information to answer the user's request. 
        The query should be concise, use relevant keywords, and exclude conversational language.
        
        Last user message: {content}
        
        Search query:
        """
        
        response = await llm_model.generate(prompt)
        query = response.strip()
        
        logger.info(f"Generated search query: {query}")
        return query
    except Exception as e:
        logger.error(f"Error generating search query: {str(e)}")
        # On error, return the last message content with context
        return _enhance_query_with_context(content)


def _enhance_query_with_context(query: str) -> str:
    """
    Enhances a search query with temporal and contextual information.
    
    Args:
        query: Original search query
        
    Returns:
        Enhanced query with date context if relevant
    """
    query_lower = query.lower()
    
    # Check if query is asking for current/recent information
    temporal_keywords = ["today", "latest", "recent", "current", "now", "this week", "news"]
    needs_date = any(keyword in query_lower for keyword in temporal_keywords)
    
    if needs_date:
        # Add current date context
        current_date = datetime.now()
        month_year = current_date.strftime("%B %Y")  # e.g., "December 2025"
        
        # If query mentions "today" or "latest", add the month/year
        if "today" in query_lower or "latest" in query_lower:
            enhanced = f"{query} {month_year}"
            logger.info(f"Enhanced query with date: '{query}' -> '{enhanced}'")
            return enhanced
    
    return query