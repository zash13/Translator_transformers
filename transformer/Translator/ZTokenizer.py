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
from typing import Dict, Optional


class SpecialToken(Enum):
    """
    enum representing special tokens with both numeric id (value)
    and string representation (token_str).
    this is how you can use it:

    token = SpecialToken.PAD
    print(token.value)       # ➜ 2
    print(token.token_str)   # ➜ "[PAD]"
    print(str(token))        # ➜ "[PAD]"
    """

    UNK = (0, "[UNK]")
    CLS = (1, "[CLS]")
    PAD = (2, "[PAD]")
    SEP = (3, "[SEP]")
    MASK = (4, "[MASK]")

    def __new__(cls, value, token_str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.token_str = token_str
        return obj

    def __str__(self):
        return self.token_str

    @classmethod
    def by_token(cls, token_str: str):
        """Find enum by token string (e.g. '[PAD]' → SpecialToken.PAD)."""
        for tok in cls:
            if tok.token_str == token_str:
                return tok
        raise ValueError(f"Unknown special token: {token_str}")

    @classmethod
    def all_tokens_str(cls):
        """Return list of token strings like ['[UNK]', '[CLS]', ...]."""
        return [tok.token_str for tok in cls]

    @classmethod
    def as_dict(cls):
        """Return mapping {token_str: id}."""
        return {tok.token_str: tok.value for tok in cls}


class BaseTokenizer(ABC):
    @classmethod
    @abstractmethod
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

    def post_process(self):
        """post prossessin happen  here text"""
        pass

    def get_vocab_size(self) -> int:
        """Get vocabulary size (optional)"""
        raise NotImplementedError


class PersianHazmTokenizer(BaseTokenizer):
    def __init__(self, vocab=None):
        self.normalizer = Normalizer()
        self.vocab = vocab or {}
        for tok in SpecialToken:
            self.vocab.setdefault(tok.token_str, tok.value)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

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
            self._build_vocab(texts, min_freq)
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

        self.vocab = {tok.token_str: tok.value for tok in SpecialToken}
        idx = max(self.vocab.values()) + 1

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
            if skip_special_tokens and tok in SpecialToken.all_tokens_str():
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

    def post_process(self):
        return super().post_process()

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
            special_tokens=SpecialToken.all_tokens_str(),
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
        ids = encoding.ids

        cls_id = self.tokenizer.token_to_id(SpecialToken.CLS.token_str)
        sep_id = self.tokenizer.token_to_id(SpecialToken.SEP.token_str)
        if add_special_tokens:
            if cls_id is None:
                cls_id = SpecialToken.CLS.value
            if sep_id is None:
                sep_id = SpecialToken.SEP.value

            if not ids or ids[0] != cls_id:
                ids = [cls_id] + ids
            if not ids or ids[-1] != sep_id:
                ids = ids + [sep_id]

            return ids
        else:
            if cls_id is not None and ids and ids[0] == cls_id:
                ids = ids[1:]
            if sep_id is not None and ids and ids[-1] == sep_id:
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

    def get_id(self, token):
        id_ = self.tokenizer.token_to_id(token)
        if id_ is None:
            id_ = self.tokenizer.token_to_id(SpecialToken.UNK.token_str)
        return id_

    def get_vocab_size(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train_by_text first.")
        try:
            return self.tokenizer.get_vocab_size()
        except Exception:
            return len(self.tokenizer.get_vocab())


class TokenizerType(Enum):
    WORDPIECE = auto()
    HAZM = auto()


# keep the enum clean
TOKENIZER_REGISTRY = {
    TokenizerType.HAZM: PersianHazmTokenizer,
    TokenizerType.WORDPIECE: EnglishTokenizer,
}


class TokenizerBuilder:
    def __init__(self):
        self._type: Optional[TokenizerType] = None
        self._params = {}
        self._tokenizer_cls = None

    @classmethod
    def _print_args(cls, items_list) -> None:
        for arg, desc in items_list:
            print(f"  - {arg}: {desc}")

    @classmethod
    def show_requirements(cls, tokenizerType: TokenizerType):
        tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizerType)
        if not tokenizer_cls:
            raise ValueError(f"Unknown tokenizer type: {tokenizerType}")
        cls._print_args(tokenizer_cls.required_args().items())

    def set_type(self, tokenizerType: TokenizerType):
        """specify which tokenizer to build"""
        if self._type != tokenizerType:
            self._params.clear()
        self._type = tokenizerType
        # this is the fluent interface , or method chaining , the core of builder pattern
        # https://dev.to/mandrewcito/fluent-interface-in-python-5b4n
        return self

    def set_params(self, **kwargs):
        if not self._type:
            raise ValueError("You must set_type  a tokenizer  first.")
        self._params.update(kwargs)
        return self

    def _choose_tokenizer(self):
        try:
            self._tokenizer_cls = TOKENIZER_REGISTRY[self._type]
        except KeyError:
            raise ValueError(f"Unknown tokenizer type: {self._type}")
        return self

    def build(self):
        """build the tokenizer with the provided parameters"""
        if not self._type:
            raise ValueError("You must set_type  a tokenizer  first.")
        self._choose_tokenizer()
        required_args = self._tokenizer_cls.required_args()
        missing_args = [arg for arg in required_args if arg not in self._params]
        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}")
        tokenizer = self._tokenizer_cls()
        tokenizer.build(**self._params)
        print(f"{self._tokenizer_cls.__name__} successfully built ")
        return tokenizer
