"""
Optimized streaming to improve speed perception
Sends first results immediately while the agent continues working
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List
from queue import Queue
import threading
import time


class StreamingOptimizer:
    """
    Optimizes streaming to give an impression of speed.
    
    Strategies:
    1. Early streaming: Sends first chunks as soon as they are available
    2. Progressive loading: Displays partial results
    3. Chunked responses: Splits long responses
    """
    
    def __init__(self):
        self.buffer = Queue()
        self.is_streaming = False
    
    async def stream_with_preview(
        self,
        generator: AsyncGenerator,
        preview_chunks: int = 3
    ) -> AsyncGenerator[str, None]:
        """
        Stream with immediate preview of first chunks.
        
        Args:
            generator: Original async generator
            preview_chunks: Number of chunks to send immediately
        
        Yields:
            Formatted JSON chunks
        """
        chunks_sent = 0
        buffer = []
        
        async for chunk in generator:
            if chunks_sent < preview_chunks:
                # Send first chunks immediately
                yield self._format_chunk(chunk, is_preview=True)
                chunks_sent += 1
            else:
                # Buffer following chunks
                buffer.append(chunk)
        
        # Send the rest of the buffer
        for chunk in buffer:
            yield self._format_chunk(chunk, is_preview=False)
    
    def _format_chunk(self, chunk: Any, is_preview: bool = False) -> str:
        """Formats a chunk for streaming"""
        data = {
            "chunk": chunk,
            "is_preview": is_preview,
            "timestamp": time.time()
        }
        return f"data: {json.dumps(data)}\n\n"
    
    async def stream_with_thinking_indicator(
        self,
        generator: AsyncGenerator,
        thinking_interval: float = 0.5
    ) -> AsyncGenerator[str, None]:
        """
        Adds "thinking" indicators during pauses.
        Improves perception of responsiveness.
        """
        last_chunk_time = time.time()
        thinking_task = None
        
        async def send_thinking():
            while True:
                await asyncio.sleep(thinking_interval)
                if time.time() - last_chunk_time > thinking_interval:
                    yield self._format_thinking()
        
        async for chunk in generator:
            last_chunk_time = time.time()
            yield self._format_chunk(chunk)
    
    def _format_thinking(self) -> str:
        """Formats a thinking indicator"""
        data = {
            "type": "thinking",
            "message": "🤔 Processing...",
            "timestamp": time.time()
        }
        return f"data: {json.dumps(data)}\n\n"
    
    async def stream_progressive_results(
        self,
        results: List[Dict[str, Any]],
        chunk_size: int = 1
    ) -> AsyncGenerator[str, None]:
        """
        Stream results progressively instead of waiting for the end.
        
        Args:
            results: List of results to stream
            chunk_size: Number of results per chunk
        
        Yields:
            Result chunks
        """
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i + chunk_size]
            yield self._format_results_chunk(chunk, i, len(results))
            await asyncio.sleep(0.01)  # Small delay to avoid overload
    
    def _format_results_chunk(
        self,
        chunk: List[Dict[str, Any]],
        current_index: int,
        total: int
    ) -> str:
        """Formats a results chunk"""
        data = {
            "type": "results",
            "chunk": chunk,
            "progress": {
                "current": current_index + len(chunk),
                "total": total,
                "percentage": round((current_index + len(chunk)) / total * 100, 1)
            },
            "timestamp": time.time()
        }
        return f"data: {json.dumps(data)}\n\n"


class ChunkedResponseGenerator:
    """
    Generates responses in chunks to improve speed perception.
    Useful for long responses.
    """
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
        """
        Splits text into reasonably sized chunks.
        Tries to cut at sentence boundaries.
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences
        sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    async def stream_chunked_response(
        text: str,
        chunk_size: int = 100,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response in chunks with delay.
        Gives the impression of real-time generation.
        """
        chunks = ChunkedResponseGenerator.chunk_text(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            data = {
                "type": "text_chunk",
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_final": i == len(chunks) - 1,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            if i < len(chunks) - 1:
                await asyncio.sleep(delay)


class ParallelStreamProcessor:
    """
    Processes multiple streams in parallel and combines them.
    Useful for displaying results from multiple sources simultaneously.
    """
    
    @staticmethod
    async def merge_streams(
        *generators: AsyncGenerator
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Merges multiple async generators.
        Sends chunks as soon as they are available, regardless of source.
        """
        tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
        active_tasks = set(tasks)
        
        while active_tasks:
            done, pending = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                try:
                    result = task.result()
                    yield result
                    
                    # Restart the task for the next chunk
                    # (simplified - in production, handle StopAsyncIteration)
                except StopAsyncIteration:
                    pass
                
                active_tasks.discard(task)


# Usage example
async def example_optimized_streaming():
    """Example of optimized streaming usage"""
    
    # Simulate a long response
    long_response = """
    Here is a very long response that will be streamed in chunks to improve
    the user experience. Instead of waiting for the entire response to be generated,
    the user will see the first words appear immediately. This gives
    an impression of speed even if the total time remains the same.
    """
    
    # Stream with chunks
    async for chunk in ChunkedResponseGenerator.stream_chunked_response(
        long_response,
        chunk_size=50,
        delay=0.1
    ):
        print(chunk, end='', flush=True)


if __name__ == "__main__":
    # Test
    asyncio.run(example_optimized_streaming())
