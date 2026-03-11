"""
Enhanced Persistent Memory Store with Ranking and Overwrite Strategy.

Features:
- Memory importance scoring (1-5)
- Keep only top 10 most useful memories per user
- Remove irrelevant/duplicate memories
- Smart memory deduplication
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import threading
import hashlib


class MemorySnippet:
    """Represents a single memory snippet with metadata."""
    
    def __init__(
        self,
        content: str,
        score: float = 3.0,
        created_at: Optional[str] = None,
        category: str = "general"
    ):
        self.content = content
        self.score = score  # 1-5 importance score
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.category = category  # "docs", "product", "video", "general"
        self.content_hash = self._compute_hash(content)
    
    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute hash for deduplication."""
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "score": self.score,
            "created_at": self.created_at,
            "category": self.category,
            "content_hash": self.content_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemorySnippet":
        snippet = cls(
            content=data["content"],
            score=data.get("score", 3.0),
            created_at=data.get("created_at"),
            category=data.get("category", "general"),
        )
        snippet.content_hash = data.get("content_hash", snippet.content_hash)
        return snippet


class MemoryStore:
    """
    Enhanced persistent memory storage with ranking.

    Structure:
    {
      "user_id_1": {
        "memories": [
          {"content": "...", "score": 4.5, "created_at": "...", "category": "product"},
          ...
        ],
        "max_memories": 10
      }
    }
    """
    
    MAX_MEMORIES_PER_USER = 10
    MIN_SCORE_THRESHOLD = 2.0
    
    def __init__(self, file_path: Optional[Path] = None, llm=None):
        if file_path is None:
            file_path = Path(__file__).parent / "user_memory.json"
        self.file_path = file_path
        self._lock = threading.Lock()
        self._store: Dict[str, Dict] = self._load()
        self._llm = llm  # Optional LLM for scoring
    
    def _load(self) -> Dict[str, Dict]:
        """Load memory store from disk."""
        if not self.file_path.exists():
            return {}
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert old format to new format
            converted = {}
            for user_id, value in data.items():
                if isinstance(value, list):
                    # Old format: list of strings
                    converted[user_id] = {
                        "memories": [
                            MemorySnippet(content=s, score=3.0).to_dict()
                            for s in value if s
                        ]
                    }
                elif isinstance(value, dict):
                    # New format
                    converted[user_id] = value
                else:
                    converted[user_id] = {"memories": []}
            
            return converted
        except Exception:
            return {}
    
    def _save(self) -> None:
        """Save memory store to disk."""
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)
    
    def _get_user_memories(self, user_id: str) -> List[MemorySnippet]:
        """Get user's memories as MemorySnippet objects."""
        user_data = self._store.get(user_id, {"memories": []})
        memories_data = user_data.get("memories", [])
        return [MemorySnippet.from_dict(m) for m in memories_data]
    
    def _set_user_memories(self, user_id: str, memories: List[MemorySnippet]):
        """Set user's memories."""
        if user_id not in self._store:
            self._store[user_id] = {"memories": []}
        self._store[user_id]["memories"] = [m.to_dict() for m in memories]
    
    def _score_memory(self, content: str, category: str = "general") -> float:
        """
        Score a memory snippet for importance (1-5).
        Uses heuristics if no LLM available.
        """
        score = 3.0  # Default score
        
        # Heuristic scoring
        content_lower = content.lower()
        
        # Boost for specific categories
        if category == "product":
            score += 0.5  # Product memories are valuable
        
        # Boost for actionable content
        action_words = ["build", "create", "implement", "design", "develop", "want", "need", "goal"]
        if any(word in content_lower for word in action_words):
            score += 0.5
        
        # Penalize generic content
        generic_phrases = ["user wants", "user is interested", "general interest"]
        if any(phrase in content_lower for phrase in generic_phrases):
            score -= 0.5
        
        # Boost for specific details
        if len(content) > 50:
            score += 0.3  # More detailed memories are better
        
        # Penalize very short memories
        if len(content) < 20:
            score -= 0.5
        
        # Clamp to 1-5 range
        return max(1.0, min(5.0, score))
    
    def _is_duplicate(self, new_content: str, existing_memories: List[MemorySnippet]) -> bool:
        """Check if memory is duplicate or very similar."""
        new_hash = MemorySnippet._compute_hash(new_content)
        new_words = set(new_content.lower().split())
        
        for mem in existing_memories:
            # Exact hash match
            if mem.content_hash == new_hash:
                return True
            
            # Similar content (>80% word overlap)
            existing_words = set(mem.content.lower().split())
            if new_words and existing_words:
                overlap = len(new_words & existing_words) / max(len(new_words), len(existing_words))
                if overlap > 0.8:
                    return True
        
        return False
    
    def _prune_memories(self, memories: List[MemorySnippet]) -> List[MemorySnippet]:
        """
        Prune memories to keep only top N by score.
        Remove low-score and duplicate memories.
        """
        # Remove low-score memories
        memories = [m for m in memories if m.score >= self.MIN_SCORE_THRESHOLD]
        
        # Sort by score (descending), then by recency
        memories.sort(key=lambda m: (m.score, m.created_at), reverse=True)
        
        # Keep only top N
        return memories[:self.MAX_MEMORIES_PER_USER]
    
    def get_memory(self, user_id: str, category: Optional[str] = None) -> str:
        """
        Return relevant memory snippets joined.
        Optionally filter by category.
        """
        with self._lock:
            memories = self._get_user_memories(user_id)
            
            if not memories:
                return ""
            
            # Filter by category if specified
            if category:
                memories = [m for m in memories if m.category == category or m.category == "general"]
            
            # Sort by score and return top memories
            memories.sort(key=lambda m: m.score, reverse=True)
            
            # Return top 5 memories joined
            top_memories = memories[:5]
            return "\n".join(m.content for m in top_memories)
    
    def get_all_memories(self, user_id: str) -> List[Dict]:
        """Get all memories with metadata for display."""
        with self._lock:
            memories = self._get_user_memories(user_id)
            memories.sort(key=lambda m: m.score, reverse=True)
            return [m.to_dict() for m in memories]
    
    def save_memory(
        self,
        user_id: str,
        snippet: str,
        category: str = "general",
        score: Optional[float] = None
    ) -> bool:
        """
        Save a memory snippet with automatic scoring and deduplication.
        Returns True if saved, False if duplicate or low quality.
        """
        if not snippet or len(snippet.strip()) < 5:
            return False
        
        snippet = snippet.strip()
        
        with self._lock:
            memories = self._get_user_memories(user_id)
            
            # Check for duplicates
            if self._is_duplicate(snippet, memories):
                return False
            
            # Score the memory
            if score is None:
                score = self._score_memory(snippet, category)
            
            # Skip low-quality memories
            if score < self.MIN_SCORE_THRESHOLD:
                return False
            
            # Create new memory snippet
            new_memory = MemorySnippet(
                content=snippet,
                score=score,
                category=category,
            )
            
            # Add to memories
            memories.append(new_memory)
            
            # Prune to keep only top memories
            memories = self._prune_memories(memories)
            
            # Save
            self._set_user_memories(user_id, memories)
            self._save()
            
            return True
    
    def update_score(self, user_id: str, content_hash: str, new_score: float):
        """Update the score of an existing memory."""
        with self._lock:
            memories = self._get_user_memories(user_id)
            
            for mem in memories:
                if mem.content_hash == content_hash:
                    mem.score = max(1.0, min(5.0, new_score))
                    break
            
            # Re-prune after score update
            memories = self._prune_memories(memories)
            self._set_user_memories(user_id, memories)
            self._save()
    
    def delete_memory(self, user_id: str, content_hash: str):
        """Delete a specific memory."""
        with self._lock:
            memories = self._get_user_memories(user_id)
            memories = [m for m in memories if m.content_hash != content_hash]
            self._set_user_memories(user_id, memories)
            self._save()
    
    def clear_user_memories(self, user_id: str):
        """Clear all memories for a user."""
        with self._lock:
            if user_id in self._store:
                self._store[user_id] = {"memories": []}
                self._save()
    
    def get_memory_stats(self, user_id: str) -> Dict:
        """Get statistics about user's memories."""
        with self._lock:
            memories = self._get_user_memories(user_id)
            
            if not memories:
                return {"count": 0, "avg_score": 0, "categories": {}}
            
            categories = {}
            for m in memories:
                categories[m.category] = categories.get(m.category, 0) + 1
            
            return {
                "count": len(memories),
                "avg_score": sum(m.score for m in memories) / len(memories),
                "categories": categories,
                "oldest": min(m.created_at for m in memories),
                "newest": max(m.created_at for m in memories),
            }

