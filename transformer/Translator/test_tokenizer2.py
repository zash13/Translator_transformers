import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List

import pandas as pd
from hazm import Normalizer, word_tokenize
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from typing import Dict


class TokenizerType(Enum):
    WORDPIECE = auto()
    HAZM = auto()


SPECIAL_TOKEN_IDS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[PAD]", "[SEP]", "[MASK]"]


class SpecialToken(Enum):
    # in .net, enums can have both a numeric value and a description (using helpers or attributes).
    # here, i'm trying to apply the same idea in python.
    # this is how you can use it:
    #
    # token = SpecialToken.PAD
    # print(token.value)       # ➜ 2
    # print(token.token_str)   # ➜ "[PAD]"
    # print(str(token))        # ➜ "[PAD]"
    UNK = (0, "[UNK]")
    CLS = (1, "[CLS]")
    PAD = (2, "[PAD]")
    SEP = (3, "[SEP]")
    MASK = (4, "[MASK]")

    def __init__(self, value, token_str):
        self._value_ = value
        self.token_str = token_str

    def __str__(self):
        return self.token_str

    @classmethod
    def by_token(cls, token_str):
        """find enum by token string."""
        for tok in cls:
            if tok.token_str == token_str:
                return tok
        raise ValueError(f"unknown special token: {token_str}")

    @classmethod
    def all_tokens(cls):
        """return list of token strings like ['[unk]'..]"""
        return [tok.token_str for tok in cls]

    @classmethod
    def as_dict(cls):
        """return mapping {token_str: id}"""
        return {tok.token_str: tok.value for tok in cls}


class BaseTokenizer(ABC):
    @classmethod
    def required_args(cls) -> Dict[str, str]:
        """return a list of required arguments for building this tokenizer"""
        pass

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError

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

        for tok in SpecialToken:
            self.vocab.setdefault(tok.token_str, tok.value)
            self.inv_vocab[tok.value] = tok.token_str

    @classmethod
    def required_args(cls):
        return {
            "texts": "a list of the sentence  , used to excute the word and create the vocab",
            "min_freq": "how aften tokenizer need to see a word to add it to vocab if it set to 2 : then each word need to seen at list 2 time",
        }

    # this may not needed , like it can simply writen like this ; return self.train_by_text(**kwargs)
    def build(self, **kwargs):
        texts = kwargs.get("texts")
        min_freq = kwargs.get("min_freq", 2)
        if texts:
            self.train_by_text(texts, min_freq)
        return self

    def train_by_text(self, texts, min_freq) -> bool:
        """
        use this method to create your tokens :
        texts: a list of the sentence  , used to excute the word and create the vocab
        min_freq : how aften tokenizer need to see a word to add it to vocab if it set to 2 : then each word need to seen at list 2 time
        """
        try:
            self.tokenizer = self._build_vocab(texts, min_freq)
            return True
        except Exception as e:
            print(f"Something wrong happened: {e}")
            return False

    def _build_vocab(self, texts, min_freq):
        normalizer = Normalizer()
        freq = {}

        for line in texts:
            tokens = word_tokenize(normalizer.normalize(line))
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

        self.vocab = {tok.token_str: tok.value for tok in SpecialToken.all_tokens()}
        idx = len(SpecialToken.all_tokens())

        for t, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            if c >= min_freq:
                self.vocab[t] = idx
                idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        normalized = self.normalizer.normalize(text)
        tokens = word_tokenize(normalized)
        return tokens

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(t, SpecialToken.UNK.value) for t in tokens]

        if add_special_tokens:
            token_ids = [SpecialToken.CLS.value] + token_ids + [SpecialToken.SEP.value]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, SpecialToken.UNK.token_str)
            if skip_special_tokens and tok in SpecialToken.all_tokens():
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def pad(self, batch_ids, max_length=None):
        max_len = max_length or max(len(seq) for seq in batch_ids)
        padded = []
        for seq in batch_ids:
            padded_seq = seq + [SpecialToken.PAD.value] * (max_len - len(seq))
            padded.append(padded_seq)
        return padded

    def get_vocab_size(self):
        return len(self.vocab)


class EnglishTokenizer(BaseTokenizer):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def required_args(cls):
        return {
            "texts": "a list of the sentence  , used to excute the word and create the vocab",
            "vocab_size": "size of your vocab !",
            "unk_token": "token that tokenizer will use for unknown word , default is SpecialToken.UNK",
        }

    def build(self, **kwargs):
        texts = kwargs.get("texts")
        vocab_size = kwargs.get("vocab_size")
        unk_token = kwargs.get("unk_token", SpecialToken.UNK.token_str)
        if texts and vocab_size:
            self.train_by_text(texts, vocab_size, unk_token)
        return self

    def train_by_text(
        self, texts, vocab_size, unk_token=SpecialToken.UNK.token_str
    ) -> bool:
        """
        use this method to create your tokens :
        texts: a list of the sentence  , used to excute the word and create the vocab
        vocab_size : size of your vocab !
        unk_token : token that tokenizer will use for unknown word , default is SpecialToken.UNK
        """
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
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=SpecialToken.all_tokens(),
        )
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
            if ids and ids[0] == SpecialToken.CLS.value:
                ids = ids[1:]
            if ids and ids[-1] == SpecialToken.SEP.value:
                ids = ids[:-1]
            return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def pad(self, batch_ids, max_length=None):
        max_len = max_length or max(len(seq) for seq in batch_ids)
        padded = [
            seq + [SpecialToken.PAD.value] * (max_len - len(seq)) for seq in batch_ids
        ]
        return padded

    def get_vocab_size(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        return self.tokenizer.get_vocab_size()


class TokenizerBuilder:
    """
    A Builder class for creating and training tokenizers.

    Example usage:

        builder = TokenizerBuilder()
        hazm_tokenizer = (
            builder
            .set_type(TokenizerType.HAZM)
            .set_training_data(texts)
            .set_min_freq(2)
            .build()
        )

        english_tokenizer = (
            builder
            .set_type(TokenizerType.WORDPIECE)
            .set_training_data(texts)
            .set_vocab_size(5000)
            .build()
        )
    """

    def __init__(self) -> None:
        pass

    def CreateToeknizer(self, tokenizerType: TokenizerType) -> BaseTokenizer:
        if tokenizerType == TokenizerType.HAZM:
            return PersianHazmTokenizer
        if tokenizerType == TokenizerType.WORDPIECE:
            return EnglishTokenizer


def create_english_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


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
