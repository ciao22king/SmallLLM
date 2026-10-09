"""
Lightweight Tokenizer for SmallLLM
Optimized for very low memory usage
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class Tokenizer:
    """
    Minimal BPE-style tokenizer for SmallLLM.
    Uses a compact vocabulary that fits in memory.
    """
    
    def __init__(self, vocab_size: int = 2048, cache_dir: str = ".smallllm_cache"):
        """
        Initialize tokenizer with lazy loading support.
        
        Args:
            vocab_size: Size of vocabulary (2048 is good for tiny models)
            cache_dir: Directory to cache vocabulary files
        """
        self.vocab_size = vocab_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Lazy-loaded components
        self._vocab: Optional[Dict[str, int]] = None
        self._reverse_vocab: Optional[Dict[int, str]] = None
        self._merges: Optional[Dict[Tuple[str, str], str]] = None
        self._pattern: Optional[re.Pattern] = None
        
        # Basic regex for splitting tokens (loaded on demand)
        self._pattern_str = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|\S"
    
    def _load_vocab(self) -> None:
        """Load or generate vocabulary on first use."""
        if self._vocab is not None:
            return
        
        vocab_file = self.cache_dir / f"vocab_{self.vocab_size}.json"
        
        if vocab_file.exists():
            # Load existing vocabulary
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._vocab = data.get('vocab', {})
                self._reverse_vocab = {v: k for k, v in self._vocab.items()}
                self._merges = data.get('merges', {})
        else:
            # Generate minimal vocabulary
            self._generate_minimal_vocab()
            # Save for future use
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'vocab': self._vocab,
                    'merges': self._merges or {}
                }, f)
    
    def _generate_minimal_vocab(self) -> None:
        """Generate a minimal vocabulary for basic English."""
        # Start with basic characters and common words
        self._vocab = {}
        
        # Add special tokens
        special_tokens = [
            '<PAD>', '<UNK>', '<SOS>', '<EOS>', '<CLS>', '<SEP>',
            '<MASK>', '[PAD]', '[UNK]', '[SOS]', '[EOS]',
        ]
        
        for i, token in enumerate(special_tokens):
            self._vocab[token] = i
        
        # Add common English characters
        chars = (
            'a b c d e f g h i j k l m n o p q r s t u v w x y z '
            'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z '
            '0 1 2 3 4 5 6 7 8 9 '
            '. , ! ? ; : " \' ( ) [ ] { } / - + = * & % $ # @ ~ ` '
            '\n \t \r'
        ).split()
        
        # Add common Italian characters
        italian_chars = 'à è é ì ò ó ù'.split()
        chars.extend(italian_chars)
        
        start_idx = len(special_tokens)
        for i, char in enumerate(chars):
            if char not in self._vocab:
                self._vocab[char] = start_idx + i
        
        # Add common words (only if vocab size allows)
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
            'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
            'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were',
            'been', 'has', 'had', 'did', 'does', 'being', 'each', 'such', 'more',
            'very', 'much', 'many', 'before', 'through', 'where', 'right', 'mean',
            'same', 'too', 'however', 'between', 'should', 'may', 'must', 'here',
            'world', 'life', 'man', 'woman', 'child', 'year', 'day', 'thing',
            # Italian words
            'ciao', 'come', 'stai', 'bene', 'grazie', 'per', 'di', 'e', 'la', 'il',
            'un', 'una', 'non', 'mi', 'ti', 'si', 'no', 'sì', 'ma', 'o', 'che',
        ]
        
        word_start = start_idx + len(chars)
        for i, word in enumerate(common_words):
            if word not in self._vocab and len(self._vocab) < self.vocab_size:
                self._vocab[word] = word_start + i
        
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self._merges = {}
    
    @property
    def vocab(self) -> Dict[str, int]:
        """Lazy-load vocabulary."""
        self._load_vocab()
        return self._vocab
    
    @property
    def reverse_vocab(self) -> Dict[int, str]:
        """Lazy-load reverse vocabulary."""
        self._load_vocab()
        return self._reverse_vocab
    
    @property
    def pattern(self) -> re.Pattern:
        """Lazy-load regex pattern."""
        if self._pattern is None:
            self._pattern = re.compile(self._pattern_str)
        return self._pattern
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        Uses greedy matching for simplicity (no BPE merges for minimal memory).
        """
        self._load_vocab()
        
        tokens = []
        text = text.lower()  # Case-insensitive for tiny model
        
        # Simple tokenization: try to match longest tokens first
        remaining = text
        while remaining:
            matched = False
            # Try to match the longest possible token
            for length in range(min(len(remaining), 20), 0, -1):
                candidate = remaining[:length]
                if candidate in self.vocab:
                    tokens.append(self.vocab[candidate])
                    remaining = remaining[length:]
                    matched = True
                    break
            
            if not matched:
                # Fall back to character-level
                char = remaining[0]
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                    remaining = remaining[1:]
                else:
                    # Use UNK token
                    unk_id = self.vocab.get('<UNK>', self.vocab.get('[UNK]', 1))
                    tokens.append(unk_id)
                    remaining = remaining[1:]
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        self._load_vocab()
        
        text = ""
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                text += self.reverse_vocab[token_id]
            else:
                text += f"<UNK_{token_id}>"
        
        return text
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_ids_list: List[List[int]]) -> List[str]:
        """Decode a batch of token IDs."""
        return [self.decode(ids) for ids in token_ids_list]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        self._load_vocab()
        return len(self.vocab)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        self._load_vocab()
        data = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'merges': self._merges or {},
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data.get('vocab_size', 2048))
        tokenizer._vocab = data.get('vocab', {})
        tokenizer._reverse_vocab = data.get('reverse_vocab', {})
        tokenizer._merges = data.get('merges', {})
        return tokenizer
