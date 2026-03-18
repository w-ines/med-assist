"""
Conversation memory management utilities for Deep Agents.
"""

from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage


class ConversationMemoryManager:
    """
    Manages conversation memory for Deep Agents.
    Currently in-memory, can be migrated to Supabase for persistence.
    """
    
    _conversations: Dict[str, List[BaseMessage]] = {}
    
    @classmethod
    def get_history(cls, conversation_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history for a given conversation_id.
        
        Args:
            conversation_id: Unique conversation identifier
            limit: Maximum number of messages to return (most recent)
        
        Returns:
            List of messages in chronological order
        """
        history = cls._conversations.get(conversation_id, [])
        if limit:
            return history[-limit:]
        return history
    
    @classmethod
    def add_messages(cls, conversation_id: str, messages: List[BaseMessage]):
        """Add messages to conversation history."""
        if conversation_id not in cls._conversations:
            cls._conversations[conversation_id] = []
        cls._conversations[conversation_id].extend(messages)
    
    @classmethod
    def clear_conversation(cls, conversation_id: str):
        """Clear all messages for a conversation."""
        if conversation_id in cls._conversations:
            del cls._conversations[conversation_id]
    
    @classmethod
    def list_conversations(cls) -> List[Dict]:
        """List all active conversations with metadata."""
        return [
            {
                "conversation_id": conv_id,
                "message_count": len(messages),
                "last_message": messages[-1].content[:100] if messages else None
            }
            for conv_id, messages in cls._conversations.items()
        ]
    
    @classmethod
    def get_stats(cls) -> Dict:
        """Get memory statistics."""
        total_messages = sum(len(msgs) for msgs in cls._conversations.values())
        return {
            "total_conversations": len(cls._conversations),
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / len(cls._conversations) if cls._conversations else 0
        }
    
    @classmethod
    def cleanup_old_conversations(cls, keep_last_n: int = 100):
        """
        Keep only the N most recent conversations (by message count as proxy for activity).
        Useful for preventing memory overflow.
        """
        if len(cls._conversations) <= keep_last_n:
            return
        
        # Sort by message count (most active conversations)
        sorted_convs = sorted(
            cls._conversations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Keep only top N
        cls._conversations = dict(sorted_convs[:keep_last_n])


# Export for easy import
__all__ = ["ConversationMemoryManager"]
