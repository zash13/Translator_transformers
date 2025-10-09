from hazm import Normalizer, word_tokenize
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import os

from abc import ABC, abstractmethod
from typing import List, Union


SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[PAD]", "[SEP]", "[MASK]"]
SPECIAL_TOKEN_IDS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs"""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        pass

    @abstractmethod
    def pad(
        self, batch_ids: List[List[int]], max_length: int = None
    ) -> List[List[int]]:
        """Pad a batch of token ID sequences"""
        pass

    def get_vocab_size(self) -> int:
        """Get vocabulary size (optional)"""
        raise NotImplementedError


class PersianHazmTokenizer(BaseTokenizer):
    def __init__(self, vocab=None):
        self.normalizer = Normalizer()
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        for tok, idx in SPECIAL_TOKEN_IDS.items():
            self.vocab.setdefault(tok, idx)
            self.inv_vocab[idx] = tok

    def tokenize(self, text):
        normalized = self.normalizer.normalize(text)
        tokens = word_tokenize(normalized)
        return tokens

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(t, SPECIAL_TOKEN_IDS["[UNK]"]) for t in tokens]
        if add_special_tokens:
            token_ids = (
                [SPECIAL_TOKEN_IDS["[CLS]"]] + token_ids + [SPECIAL_TOKEN_IDS["[SEP]"]]
            )
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "[UNK]")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def pad(self, batch_ids, max_length=None):
        max_len = max_length or max(len(seq) for seq in batch_ids)
        padded = []
        for seq in batch_ids:
            padded_seq = seq + [SPECIAL_TOKEN_IDS["[PAD]"]] * (max_len - len(seq))
            padded.append(padded_seq)
        return padded

    def get_vocab_size(self):
        return len(self.vocab)


class EnglishTokenizer(BaseTokenizer):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def train_by_text(self, texts, vocab_size, unk_token="[UNK]") -> bool:
        try:
            self.tokenizer = self._create_english_tokenizer(
                texts, vocab_size, unk_token
            )
            return True
        except Exception as e:
            print(f"Something wrong happened: {e}")
            return False

    def _create_english_tokenizer(self, texts, vocab_size, unk_token):
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return tokenizer

    def tokenize(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def encode(self, text, add_special_tokens=True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        encoding = self.tokenizer.encode(text)
        if add_special_tokens:
            return encoding.ids
        else:
            ids = encoding.ids
            if ids and ids[0] == SPECIAL_TOKEN_IDS["[CLS]"]:
                ids = ids[1:]
            if ids and ids[-1] == SPECIAL_TOKEN_IDS["[SEP]"]:
                ids = ids[:-1]
            return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def pad(self, batch_ids, max_length=None):
        if max_length is None:
            max_len = max(len(seq) for seq in batch_ids)
        else:
            max_len = max_length

        padded = []
        for seq in batch_ids:
            padded_seq = seq + [SPECIAL_TOKEN_IDS["[PAD]"]] * (max_len - len(seq))
            padded.append(padded_seq)
        return padded

    def get_vocab_size(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        return self.tokenizer.get_vocab_size()


def create_english_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def build_persian_vocab(texts, min_freq=2):
    normalizer = Normalizer()
    freq = {}
    for line in texts:
        tokens = word_tokenize(normalizer.normalize(line))
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

    vocab = dict(SPECIAL_TOKEN_IDS)
    idx = len(SPECIAL_TOKEN_IDS)
    for t, c in sorted(freq.items(), key=lambda x: -x[1]):
        if c >= min_freq:
            vocab[t] = idx
            idx += 1
    return vocab


src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")


def read_data(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f:
        src_data = [l.strip() for l in f if l.strip()]
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_data = [l.strip() for l in f if l.strip()]
    return pd.DataFrame({"src": src_data, "tgt": tgt_data})


print("loading TEP data...")
full_data = read_data(src_file, tgt_file)

print("Training English tokenizer...")
eng_tokenizer = create_english_tokenizer(full_data["src"], vocab_size=30000)

print("Building Persian vocab...")
persian_vocab = build_persian_vocab(full_data["tgt"])

print("Initializing Persian tokenizer...")
persian_tokenizer = PersianHazmTokenizer(vocab=persian_vocab)


sample_persian = "من به مدرسه می‌روم."
sample_english = "I am going to school."

persian_ids = persian_tokenizer.encode(sample_persian)
print("Persian IDs:", persian_ids)
print("Decoded Persian:", persian_tokenizer.decode(persian_ids))

persian_ids = persian_tokenizer.encode(sample_persian)
print("Original Persian IDs:", persian_ids)
print("Original length:", len(persian_ids))


padded_ids = persian_tokenizer.pad([persian_ids], max_length=30)[0]
print("Padded Persian IDs:", padded_ids)
print("Padded length:", len(padded_ids))
print("Decoded Persian (with padding):", persian_tokenizer.decode(padded_ids))
