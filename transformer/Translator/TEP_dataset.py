import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import sacrebleu
import keras
from tqdm import tqdm
from tokenizers import Tokenizer


gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs available:", len(gpus))

MAX_SEQUENCE_LENGTH = 30
# this this 2 to zero if you want to use max vocab size
INPUT_VOCAB_SIZE = 90000
TARGET_VOCAB_SIZE = 120000
BATCH_SIZE = 64
EPOCHS = 3
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

import keras
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack
from TokenEmbedding import TokenEmbedding
from PositionalEncoding import positional_encoding
import tensorflow as tf


def normalize_texts(texts, lower=True, strip=True):
    """
    Normalize a list of input strings before tokenization.
    - lower: convert to lowercase
    - strip: remove extra whitespace
    """
    if isinstance(texts, str):
        texts = [texts]  # support single string

    normalized = []
    for t in texts:
        if lower:
            t = t.lower()
        if strip:
            t = " ".join(t.split())  # remove duplicate spaces
        normalized.append(t)

    return normalized if len(normalized) > 1 else normalized[0]


class Translator(keras.Model):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
        pad_id,
        rate=0.1,
    ):
        super().__init__()
        self.encoder_embedding = TokenEmbedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = TokenEmbedding(target_vocab_size, embedding_dim)
        self.positional_encoding = positional_encoding(
            max_sequence_length, embedding_dim
        )
        self.encoder = EncoderStack(
            num_layers, embedding_dim, num_heads, feed_forward_dim, rate
        )
        self.decoder = DecoderStack(
            num_layers, embedding_dim, num_heads, feed_forward_dim, rate
        )
        self.final_layer = keras.layers.Dense(target_vocab_size)
        self.pad_id = pad_id
        self.max_sequence_length = max_sequence_length

    def build(self, input_shape):
        # input_shape: [(batch, enc_seq_len), (batch, dec_seq_len)]
        # This model performs both training and prediction using batches, which is faster as you know.
        # That's why I accept the input in this format.
        enc_input_shape, dec_input_shape = input_shape
        self.encoder_embedding.build(enc_input_shape)
        self.decoder_embedding.build(dec_input_shape)
        self.encoder.build((None, None, self.encoder_embedding.embedding_dim))
        self.decoder.build((None, None, self.decoder_embedding.embedding_dim))
        self.final_layer.build((None, None, self.decoder_embedding.embedding_dim))
        self.built = True

    def call(self, inputs, training=True):
        enc_inputs, dec_inputs = inputs  # input = [encoder_input, decoder_input]

        enc_padding_mask = self.create_padding_mask(enc_inputs)
        dec_padding_mask = self.create_padding_mask(dec_inputs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(dec_inputs)[1])
        dec_self_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        enc_emb = self.encoder_embedding(enc_inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=training, padding_mask=enc_padding_mask
        )

        # very important: providing this variable helps the model understand its current position in the sequence.
        # it gives the model awareness of the past, how far it has progressed, and help it with next.
        dec_emb = self.decoder_embedding(dec_inputs)
        dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
        dec_output = self.decoder(
            dec_emb,
            enc_output,
            training=training,
            look_ahead_mask=dec_self_mask,
            padding_mask=enc_padding_mask,
        )

        final_output = self.final_layer(dec_output)
        return final_output

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, self.pad_id), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def generate(self, inputs, max_sequence_length, start_token, end_token):
        batch_size = tf.shape(inputs)[0]

        # Encode once
        enc_padding_mask = self.create_padding_mask(inputs)
        enc_emb = self.encoder_embedding(inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=False, padding_mask=enc_padding_mask
        )

        # Initialize with <BOS>
        decoded = tf.fill([batch_size, 1], start_token)
        finished = tf.zeros([batch_size], dtype=tf.bool)

        for _ in tf.range(max_sequence_length - 1):
            look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoded)[1])
            dec_emb = self.decoder_embedding(decoded)
            dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
            dec_output = self.decoder(
                dec_emb,
                enc_output,
                training=False,
                look_ahead_mask=look_ahead_mask,
                padding_mask=enc_padding_mask,
            )

            # Take last step logits
            last_token_output = dec_output[:, -1, :]
            logits = self.final_layer(last_token_output)

            probs = tf.nn.softmax(logits[0]).numpy()
            top_ids = np.argsort(probs)[::-1][:10]
            print("Top 10 tokens:", [(id, probs[id]) for id in top_ids])
            probs = tf.nn.softmax(logits / 1.2, axis=-1)  # temperature
            predicted_ids = tf.random.categorical(tf.math.log(probs), num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)
            predicted_ids = tf.cast(predicted_ids, decoded.dtype)
            predicted_ids = tf.expand_dims(predicted_ids, 1)
            decoded = tf.concat([decoded, predicted_ids], axis=1)

            # Mark finished sequences
            finished = finished | tf.equal(predicted_ids[:, 0], end_token)
            if tf.reduce_all(finished):
                break

        return decoded

    def generate2(self, inputs, max_sequence_length, start_token, end_token):
        """
        GREADY DECODING LOOP FOR PREDICTIONS
            args :
                inputs = encoder outputs
                max_sequence_length = maximum length of the generated sequence ( incude the start and end and stuff )
                start_token = integer id of the start of seq , like 1 for <sos>
                end_token = integer id of the end of seq , like 2 for <eos>

            Returns:
                Generated sequence tensor of shape (batch, max_len).
        """
        # here's what happens: first, we calculate all the encoder outputs.
        # then, we iterate over the sequence length to perform decoding.
        # for example, if we have 10 sentences, each with length 30, we start from position 0 to 29 (30 - 1 , which is start ) .
        # at each step, we decode the current token for all 10 sentences simultaneously.
        # it's like moving vertically through the batch instead of horizontally (sentence by sentence).
        #
        #
        # calculate encoder outputs
        batch_size = tf.shape(inputs)[0]
        enc_padding_mask = self.create_padding_mask(inputs)
        enc_emb = self.encoder_embedding(inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=False, padding_mask=enc_padding_mask
        )
        # initialize decoded sequence with start token like bunch of tensors with start token [[1] , [1] , ...]
        decoded = tf.ones((batch_size, 1), dtype=tf.int32) * start_token
        finished = tf.zeros((batch_size,), dtype=tf.bool)

        for i in tf.range(max_sequence_length - 1):
            dec_padding_mask = self.create_padding_mask(decoded)
            look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoded)[1])
            dec_self_mask = tf.maximum(dec_padding_mask, look_ahead_mask)
            dec_emb = self.decoder_embedding(decoded)
            dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
            dec_output = self.decoder(
                dec_emb,
                enc_output,
                training=False,
                look_ahead_mask=dec_self_mask,
                padding_mask=enc_padding_mask,
            )
            # w
            last_token_output = dec_output[:, -1, :]
            logits = self.final_layer(last_token_output)
            # the use of argmax here is incorrect. there should be a parameter like π or α that represents
            # the probability or rate of choosing either the argmax or a lower-level alternative.
            # sometimes, there are other words similar to the argmax choice, but the model avoids selecting them.
            # the model can get stuck on a word due to poor training data or a bad input sequence,
            # or for other reasons.
            predicted_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predicted_ids = tf.expand_dims(predicted_ids, 1)  # (batch_size, 1)

            predicted_ids = tf.where(finished[:, tf.newaxis], 0, predicted_ids)
            decoded = tf.concat([decoded, predicted_ids], axis=1)

            finished = finished | (predicted_ids[:, 0] == end_token)
            if tf.reduce_all(finished):
                break
        return decoded


def read_data(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f:
        src_data = f.readlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_data = f.readlines()
    data = pd.DataFrame(
        {
            "src": [line.strip() for line in src_data],
            "tgt": [line.strip() for line in tgt_data],
        }
    )
    data = data[data["src"].str.len() > 0]
    data = data[data["tgt"].str.len() > 0]
    return data


def calculate_vocab_coverage(
    full_data, input_vocab_size, target_vocab_size, lower=True, print_output=True
):
    src_vocab = set()
    for sentence in full_data["src"]:
        if lower:
            sentence = sentence.lower()
        src_vocab.update(sentence.split())  # Word-level
    tgt_vocab = set()
    for sentence in full_data["tgt"]:
        if lower:
            sentence = sentence.lower()
        tgt_vocab.update(sentence.split())
    src_coverage = min(input_vocab_size / len(src_vocab), 1.0) * 100
    tgt_coverage = min(target_vocab_size / len(tgt_vocab), 1.0) * 100
    if print_output:
        print(f"Input (English) word vocab size: {len(src_vocab)}")
        print(f"Target (Persian) word vocab size: {len(tgt_vocab)}")
        print(f"Input tokenizer covers ~{src_coverage:.2f}% of words")
        print(f"Target tokenizer covers ~{tgt_coverage:.2f}% of words")
    # Real word-level size (for reference)
    return len(src_vocab), len(tgt_vocab), src_coverage, tgt_coverage


def create_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def test_preprocess_data(data, src_tokenizer, tgt_tokenizer, max_length):
    pad_id_src = src_tokenizer.token_to_id("[PAD]")
    pad_id_tgt = tgt_tokenizer.token_to_id("[PAD]")
    bos_id_src = src_tokenizer.token_to_id("[BOS]")
    eos_id_src = src_tokenizer.token_to_id("[EOS]")
    bos_id_tgt = tgt_tokenizer.token_to_id("[BOS]")
    eos_id_tgt = tgt_tokenizer.token_to_id("[EOS]")

    with open("preprocess_log.txt", "w") as log:
        log.write(
            f"Source PAD: {pad_id_src}, Target PAD: {pad_id_tgt}\n"
            f"Source BOS: {bos_id_src}, Source EOS: {eos_id_src}\n"
            f"Target BOS: {bos_id_tgt}, Target EOS: {eos_id_tgt}\n\n"
        )

        tf_dataset = preprocess_data(
            data,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_length=max_length,
        )

        results = []
        all_good = True

        for i, item in enumerate(
            tqdm(tf_dataset, desc="Testing samples", unit="sample")
        ):
            src, tgt_in = item[0]
            tgt_out = item[1]

            src_np = src.numpy()
            tgt_in_np = tgt_in.numpy()
            tgt_out_np = tgt_out.numpy()

            seq_len_src = next(
                (
                    ii + 1
                    for ii in reversed(range(len(src_np)))
                    if src_np[ii] != pad_id_src
                ),
                0,
            )
            seq_len_tgt = next(
                (
                    ii + 1
                    for ii in reversed(range(len(tgt_in_np)))
                    if tgt_in_np[ii] != pad_id_tgt
                ),
                0,
            )

            checks = {
                "src_length": len(src_np) == max_length,
                "tgt_in_length": len(tgt_in_np) == max_length - 1,
                "tgt_out_length": len(tgt_out_np) == max_length - 1,
                "src_starts_bos": src_np[0] == bos_id_src,
                "src_ends_eos": seq_len_src > 0
                and src_np[seq_len_src - 1] == eos_id_src,
                "src_pads_after": all(
                    src_np[j] == pad_id_src for j in range(seq_len_src, len(src_np))
                ),
                "tgt_in_starts_bos": tgt_in_np[0] == bos_id_tgt,
                "tgt_in_ends_no_eos": seq_len_tgt > 0
                and tgt_in_np[seq_len_tgt - 1] != eos_id_tgt,
                "tgt_out_ends_eos_pos": seq_len_tgt > 0
                and tgt_out_np[seq_len_tgt - 1] == eos_id_tgt,
                "tgt_out_pads_after": all(
                    tgt_out_np[j] == pad_id_tgt
                    for j in range(seq_len_tgt, len(tgt_out_np))
                ),
                "tgt_shifted": seq_len_tgt <= 1
                or all(
                    tgt_out_np[j] == tgt_in_np[j + 1] for j in range(seq_len_tgt - 1)
                ),
                "tgt_out_starts_no_bos": tgt_out_np[0] != bos_id_tgt
                if seq_len_tgt > 1
                else True,
            }

            is_good = all(checks.values())
            all_good = all_good and is_good
            results.append(
                {
                    "index": i,
                    "good": is_good,
                    "checks": checks,
                    "src": src_np.tolist(),
                    "tgt_in": tgt_in_np.tolist(),
                    "tgt_out": tgt_out_np.tolist(),
                }
            )

            log.write(f"Item {i}: Good={is_good}\n")
            log.write(f"  src: {src_np.tolist()}\n")
            log.write(f"  tgt_in: {tgt_in_np.tolist()}\n")
            log.write(f"  tgt_out: {tgt_out_np.tolist()}\n")
            log.write(f"  seq_len_src: {seq_len_src}, seq_len_tgt: {seq_len_tgt}\n")
            log.write(f"  Checks: {checks}\n\n")

        log.write(f"Overall: All good = {all_good}\n")
        if not all_good:
            bad_items = [r for r in results if not r["good"]]
            log.write(f"Bad items: {len(bad_items)}\n")
            for bad in bad_items:
                failed_checks = {k: v for k, v in bad["checks"].items() if not v}
                log.write(f"  Item {bad['index']}: Failed checks = {failed_checks}\n")

    return results, all_good


def preprocess_examples(
    src_texts,
    tgt_texts=None,
    src_tokenizer=None,
    tgt_tokenizer=None,
    max_length=30,
    training=True,
):
    """
    this is a generate method to preprocess the data
    preprocesses source (and optionally target) texts into padded token ids.

    args:
        src_texts (str or list[str]): source sentence(s).
        tgt_texts (str or list[str], optional): target sentence(s). required in training.
        training (bool): if true → returns (src, tgt_in, tgt_out).
                         if false → returns only (src,).

    returns:
        if training=true:
            list of (src, tgt_in, tgt_out) arrays
        if training=false:
            list of src arrays
    """

    if isinstance(src_texts, str):
        src_texts = [src_texts]
    if tgt_texts is not None and isinstance(tgt_texts, str):
        tgt_texts = [tgt_texts]

    pad_id_src = src_tokenizer.token_to_id("[PAD]")
    pad_id_tgt = tgt_tokenizer.token_to_id("[PAD]")
    bos_id_src = src_tokenizer.token_to_id("[BOS]")
    eos_id_src = src_tokenizer.token_to_id("[EOS]")
    bos_id_tgt = tgt_tokenizer.token_to_id("[BOS]")
    eos_id_tgt = tgt_tokenizer.token_to_id("[EOS]")

    results = []

    for i, src_text in enumerate(src_texts):
        src_ids = src_tokenizer.encode(src_text).ids[: max_length - 2]
        src_seq = [bos_id_src] + src_ids + [eos_id_src]
        src_seq = src_seq[:max_length] + [pad_id_src] * (max_length - len(src_seq))

        if training:
            tgt_text = tgt_texts[i]
            tgt_ids = tgt_tokenizer.encode(tgt_text).ids[: max_length - 2]
            tgt_seq = [bos_id_tgt] + tgt_ids + [eos_id_tgt]

            # Shifted
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
                tf.TensorSpec(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32),
                tf.TensorSpec(shape=(MAX_SEQUENCE_LENGTH - 1,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(MAX_SEQUENCE_LENGTH - 1,), dtype=tf.int32),
        ),
    )
    return train_dataset


def evaluate_bleu(model, dataset, src_tokenizer, tgt_tokenizer, max_length=30):
    refs, hyps = [], []
    for (src, _), tgt_out in dataset:
        bos_id = src_tokenizer.token_to_id("[BOS]")
        eos_id = src_tokenizer.token_to_id("[EOS]")
        pred = model.generate(src, 30, bos_id, eos_id)
        print("Raw prediction IDs:", pred[0].numpy())
        print("Decoded text:", tgt_tokenizer.decode(pred[0].numpy()))
        pred = model.generate(
            src,
            max_length,
            start_token=tgt_tokenizer.token_to_id("[BOS]"),
            end_token=tgt_tokenizer.token_to_id("[EOS]"),
        )
        for p, t in zip(pred.numpy(), tgt_out.numpy()):
            p = [id for id in p if id != tgt_tokenizer.token_to_id("[PAD]")]
            t = [id for id in t if id != tgt_tokenizer.token_to_id("[PAD]")]
            hyp = tgt_tokenizer.decode(p[1:])  # Skip [BOS]
            ref = tgt_tokenizer.decode(t)
            hyps.append(hyp)
            refs.append([ref])  # sacrebleu expects list of refs
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score


MOCK_SENTENCES = [
    {"english": "Hello, how are you?", "persian": "سلام، حالت چطوره؟"},
    {"english": "I am fine, thank you.", "persian": "من خوبم، ممنون."},
    {"english": "What is your name?", "persian": "اسمت چیه؟"},
    {"english": "I live in Tehran.", "persian": "من در تهران زندگی می‌کنم."},
    {"english": "The weather is nice today.", "persian": "هوا امروز خوبه."},
    {"english": "I like reading books.", "persian": "من کتاب خوندن رو دوست دارم."},
    {"english": "See you later.", "persian": "بعداً می‌بینمت."},
    {"english": "Good morning!", "persian": "صبح بخیر!"},
    {"english": "I need help.", "persian": "من به کمک نیاز دارم."},
    {
        "english": "Goodbye, have a good day.",
        "persian": "خداحافظ، روز خوبی داشته باشی.",
    },
]


def test_mock_predictions2(
    model, src_tokenizer, tgt_tokenizer, mock_sentences=MOCK_SENTENCES
):
    """
    Debug preprocessing: show original sentence, token IDs, and decoded sentence.
    """
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")

    src_texts = [s["english"] for s in mock_sentences]
    src_texts = normalize_texts(src_texts, lower=True)  # apply lowercase normalization

    src_seqs = preprocess_examples(
        src_texts=src_texts,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
        training=False,
    )

    for i, seq in enumerate(src_seqs):
        english_text = mock_sentences[i]["english"]

        # Filter out PADs so decode is clean
        valid_ids = [id for id in seq if id not in [pad_id]]

        # Decode with tokenizer
        decoded_text = src_tokenizer.decode(valid_ids)

        print("\n---")
        print(f"Original: {english_text}")
        print(f"Token IDs: {valid_ids}")
        print(f"Decoded : {decoded_text}")


def test_mock_predictions(
    model, src_tokenizer, tgt_tokenizer, mock_sentences=MOCK_SENTENCES
):
    """
    Tests model predictions on a hardcoded list of mock English sentences.
    Prints a table: English Input | Expected Persian | Model Prediction
    """
    results = []

    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    src_texts = [s["english"] for s in mock_sentences]
    src_texts = normalize_texts(src_texts, lower=True)
    src_seqs = preprocess_examples(
        src_texts=[s["english"] for s in mock_sentences],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
        training=False,
    )

    # Iterate over mock sentences and predictions
    for i, src_seq in enumerate(src_seqs):
        english_text = mock_sentences[i]["english"]
        expected_persian = mock_sentences[i]["persian"]

        # Wrap single src_seq into a batch of size 1
        enc_input = tf.constant([src_seq])

        # Run model generation
        pred_tokens = model.generate(enc_input, MAX_SEQUENCE_LENGTH, bos_id, eos_id)

        # Decode prediction (remove BOS/EOS/PAD)
        pred_ids = pred_tokens[0].numpy()
        pred_text = tgt_tokenizer.decode(
            [id for id in pred_ids if id not in [bos_id, eos_id, pad_id]]
        )

        results.append(
            {
                "English Input": english_text,
                "Expected Persian": expected_persian,
                "Model Prediction": pred_text,
            }
        )

    # Pretty print results
    results_df = pd.DataFrame(results)
    print("\n=== Mock Sentence Predictions ===")
    print(results_df.to_string(index=False, max_colwidth=40))


def main():
    num_layers = 2
    embedding_dim = 128
    num_heads = 4
    feed_forward_dim = 512
    input_vocab_size = INPUT_VOCAB_SIZE
    target_vocab_size = TARGET_VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    # you can ether set this variable , or let it be maximum of waht we have in dataset
    max_train_samples = 0
    val_samples_ratio = 0.2

    src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("loading TEP data...")
    full_data = read_data(src_file, tgt_file)
    total_sampels = len(full_data)

    if max_train_samples == 0:
        max_train_samples = int(total_sampels * (1 - val_samples_ratio))
        max_val_samples = int(total_sampels * (val_samples_ratio))
    else:
        max_val_samples = int(
            max_train_samples * val_samples_ratio,
        )

    train_data = full_data.sample(n=max_train_samples, random_state=42)
    val_data = full_data.drop(train_data.index).sample(
        n=max_val_samples, random_state=42
    )
    print(
        f"total samples : {total_sampels} train samples :{max_train_samples} validation sampels :{max_val_samples} validation ratio: {val_samples_ratio} "
    )

    print("Vocabulary information:")
    src_vocab_size, tgt_vocab_size, _, _ = calculate_vocab_coverage(
        full_data, input_vocab_size, target_vocab_size, lower=True, print_output=False
    )
    input_vocab_size = src_vocab_size if input_vocab_size == 0 else input_vocab_size
    target_vocab_size = tgt_vocab_size if target_vocab_size == 0 else target_vocab_size
    src_vocab_size, tgt_vocab_size, _, _ = calculate_vocab_coverage(
        full_data, input_vocab_size, target_vocab_size, lower=True, print_output=True
    )
    print("training tokenizers...")
    train_data["src"] = normalize_texts(train_data["src"], lower=True)
    val_data["src"] = normalize_texts(val_data["src"], lower=True)
    src_tokenizer = create_tokenizer(train_data["src"], INPUT_VOCAB_SIZE)
    tgt_tokenizer = create_tokenizer(train_data["tgt"], TARGET_VOCAB_SIZE)

    print("preprocessing data...")
    train_dataset = preprocess_data(
        train_data, src_tokenizer, tgt_tokenizer, max_sequence_length
    )
    val_dataset = preprocess_data(
        val_data, src_tokenizer, tgt_tokenizer, max_sequence_length
    )

    train_dataset = (
        train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("initializing model...")
    model = Translator(
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
        pad_id=tgt_tokenizer.token_to_id("[PAD]"),
    )
    model.build([(None, max_sequence_length), (None, max_sequence_length - 1)])

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("Training model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
    test_mock_predictions(model, src_tokenizer, tgt_tokenizer)
    print("Evaluating BLEU score...")
    bleu_score = evaluate_bleu(model, val_dataset, src_tokenizer, tgt_tokenizer)
    print(f"Validation BLEU: {bleu_score:.2f}")

    model.save_weights("translator_model_architecture.weights.h5")


def main2():
    input_vocab_size = INPUT_VOCAB_SIZE
    target_vocab_size = TARGET_VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    # you can ether set this variable , or let it be maximum of waht we have in dataset
    max_train_samples = 0
    val_samples_ratio = 0.2

    src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("loading TEP data...")
    full_data = read_data(src_file, tgt_file)
    total_sampels = len(full_data)

    if max_train_samples == 0:
        max_train_samples = int(total_sampels * (1 - val_samples_ratio))
        max_val_samples = int(total_sampels * (val_samples_ratio))
    else:
        max_val_samples = int(
            max_train_samples * val_samples_ratio,
        )

    train_data = full_data.sample(n=max_train_samples, random_state=42)
    val_data = full_data.drop(train_data.index).sample(
        n=max_val_samples, random_state=42
    )
    print(
        f"total samples : {total_sampels} train samples :{max_train_samples} validation sampels :{max_val_samples} validation ratio: {val_samples_ratio} "
    )

    print("Vocabulary information:")
    src_vocab_size, tgt_vocab_size, _, _ = calculate_vocab_coverage(
        full_data, input_vocab_size, target_vocab_size, lower=True, print_output=False
    )
    input_vocab_size = src_vocab_size if input_vocab_size == 0 else input_vocab_size
    target_vocab_size = tgt_vocab_size if target_vocab_size == 0 else target_vocab_size
    src_vocab_size, tgt_vocab_size, _, _ = calculate_vocab_coverage(
        full_data, input_vocab_size, target_vocab_size, lower=True, print_output=True
    )
    train_data["src"] = normalize_texts(train_data["src"], lower=True)
    val_data["src"] = normalize_texts(val_data["src"], lower=True)
    print("training tokenizers...")
    src_tokenizer = create_tokenizer(train_data["src"], INPUT_VOCAB_SIZE)
    tgt_tokenizer = create_tokenizer(train_data["tgt"], TARGET_VOCAB_SIZE)

    print("preprocessing data...")

    test_mock_predictions2(None, src_tokenizer, tgt_tokenizer)


# train_dataset, all_good = test_preprocess_data( train_data, src_tokenizer, tgt_tokenizer, max_sequence_length)


def interactive_translate(model, src_tokenizer, tgt_tokenizer):
    """
    Interactive mode: User inputs English sentence, model translates to Persian.
    Type 'quit' to exit.
    """
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")

    print("\n=== Interactive Translation Mode ===")
    print("Enter an English sentence (or 'quit' to exit):")

    while True:
        english_text = input("> ").strip()
        if english_text.lower() == "quit":
            print("Exiting interactive mode.")
            break

        if not english_text:
            print("Please enter a valid sentence.")
            continue

        # Tokenize English for encoder input (add [BOS]/[EOS], pad)
        enc_tokens = src_tokenizer.encode(english_text).ids
        enc_input = (
            [src_tokenizer.token_to_id("[BOS]")]
            + enc_tokens
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        enc_input = enc_input[:MAX_SEQUENCE_LENGTH] + [
            src_tokenizer.token_to_id("[PAD]")
        ] * (MAX_SEQUENCE_LENGTH - len(enc_input))
        enc_input = tf.constant([enc_input])  # Batch of 1

        # Predict Persian tokens
        pred_tokens = model.generate(enc_input, MAX_SEQUENCE_LENGTH, bos_id, eos_id)

        # Decode prediction (remove special tokens)
        pred_ids = pred_tokens[0].numpy()  # First batch item
        pred_text = tgt_tokenizer.decode(
            [id for id in pred_ids if id not in [bos_id, eos_id, pad_id]]
        )

        print(f"Translation: {pred_text}\n")


if __name__ == "__main__":
    main()
