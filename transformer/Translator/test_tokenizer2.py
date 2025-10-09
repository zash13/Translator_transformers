import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import pre_tokenizers
from hazm import Normalizer, word_tokenize

# Special tokens dictionary (excluding [UNK] which is handled in trainer)
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def add_special_tokens(tokenizer):
    """Manually add special tokens to the tokenizer's vocab after training."""
    vocab = tokenizer.model.vocab
    current_size = len(vocab)
    for i, tok in enumerate(SPECIAL_TOKENS, start=current_size):
        vocab[tok] = i
    # Note: The effective vocab size now exceeds the trained vocab_size by len(SPECIAL_TOKENS)


def create_english_tokenizer(texts, vocab_size):
    """Create tokenizer for English/Other languages using WordPiece with Whitespace pre-tokenizer."""
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    add_special_tokens(tokenizer)
    return tokenizer


def create_persian_tokenizer(texts, vocab_size):
    """Create tokenizer for Persian using WordPiece with Hazm pre-tokenizer."""

    def pre_tokenize(text: str) -> list[str]:
        normalizer = Normalizer()
        normalized = normalizer.normalize(text)
        return word_tokenize(normalized)

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Callable(pre_tokenize)
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    add_special_tokens(tokenizer)
    return tokenizer


def preprocess_examples(
    src_texts,
    tgt_texts=None,
    src_tokenizer=None,
    tgt_tokenizer=None,
    max_length=30,
    training=True,
):
    """Universal preprocessing — works for tokenizers library."""
    if isinstance(src_texts, str):
        src_texts = [src_texts]
    if tgt_texts is not None and isinstance(tgt_texts, str):
        tgt_texts = [tgt_texts]

    results = []

    def get_id(tokenizer, token):
        if hasattr(tokenizer, "token_to_id"):
            id_ = tokenizer.token_to_id(token)
        else:
            id_ = None
        if id_ is None:
            # fallback to tokenizer.unk_token_id or 0
            id_ = getattr(tokenizer, "unk_token_id", 0)
        return id_

    # Get special token IDs manually
    pad_id_src = get_id(src_tokenizer, "[PAD]")
    cls_id_src = get_id(src_tokenizer, "[CLS]")
    sep_id_src = get_id(src_tokenizer, "[SEP]")

    pad_id_tgt = get_id(tgt_tokenizer, "[PAD]")
    cls_id_tgt = get_id(tgt_tokenizer, "[CLS]")
    sep_id_tgt = get_id(tgt_tokenizer, "[SEP]")

    print(
        f"Source special IDs - PAD: {pad_id_src}, CLS: {cls_id_src}, SEP: {sep_id_src}"
    )
    print(
        f"Target special IDs - PAD: {pad_id_tgt}, CLS: {cls_id_tgt}, SEP: {sep_id_tgt}"
    )

    for i, src_text in enumerate(src_texts):
        # === SOURCE ===
        # Tokenize without special tokens
        encoded = src_tokenizer.encode(src_text)
        src_ids = encoded.ids if hasattr(encoded, "ids") else encoded

        src_ids = src_ids[: max_length - 2]  # Reserve space for CLS and SEP
        src_seq = [cls_id_src] + src_ids + [sep_id_src]
        src_seq = src_seq[:max_length] + [pad_id_src] * (max_length - len(src_seq))

        if training and tgt_texts is not None:
            tgt_text = tgt_texts[i]

            # === TARGET ===
            # Tokenize without special tokens
            encoded = tgt_tokenizer.encode(tgt_text)
            tgt_ids = encoded.ids if hasattr(encoded, "ids") else encoded

            tgt_ids = tgt_ids[: max_length - 2]
            tgt_seq = [cls_id_tgt] + tgt_ids + [sep_id_tgt]

            tgt_in = tgt_seq[:-1]
            tgt_out = tgt_seq[1:]

            tgt_in_len = max_length - 1
            tgt_in = tgt_in[:tgt_in_len] + [pad_id_tgt] * (tgt_in_len - len(tgt_in))
            tgt_out = tgt_out[:tgt_in_len] + [pad_id_tgt] * (tgt_in_len - len(tgt_out))

            results.append(
                (
                    np.array(src_seq, dtype=np.int32),
                    np.array(tgt_in, dtype=np.int32),
                    np.array(tgt_out, dtype=np.int32),
                )
            )
        else:
            results.append(np.array(src_seq, dtype=np.int32))

    return results


def preprocess_data(data, src_tokenizer, tgt_tokenizer, max_length=30):
    dataset = preprocess_examples(
        src_texts=data["src"].tolist(),
        tgt_texts=data["tgt"].tolist(),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=max_length,
        training=True,
    )

    # Turn into tf.data.Dataset
    def gen():
        for src, tin, tout in dataset:
            yield ((src, tin), tout)

    train_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(max_length - 1,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(max_length - 1,), dtype=tf.int32),
        ),
    )
    return train_dataset


if __name__ == "__main__":
    # load tokenizers (only used for subword tokenization and convert_tokens_to_ids)

    # Example texts
    src = "سلام، چطور هستید؟ امیدوارم روز خوبی داشته باشید"
    tgt = "Hello, how are you? Hope you have a great day!"

    # Preprocess to arrays
    dataset_list = preprocess_examples(
        src_texts=[src],
        tgt_texts=[tgt],
        src_tokenizer=persian_tokenizer,
        tgt_tokenizer=english_tokenizer,
        max_length=20,
        src_lang="fa",
        tgt_lang="en",
        training=True,
    )

    print("\nPreprocessed arrays (src, tgt_in, tgt_out):")
    for src_arr, tin_arr, tout_arr in dataset_list:
        print("src ids:", src_arr.tolist())
        print("tgt_in ids:", tin_arr.tolist())
        print("tgt_out ids:", tout_arr.tolist())

    # Optionally: convert to tf.data.Dataset
    train_ds = create_tf_dataset(dataset_list, max_length=20, batch_size=1)
    print("\nCreated tf.data.Dataset with 1 batch (example).")
