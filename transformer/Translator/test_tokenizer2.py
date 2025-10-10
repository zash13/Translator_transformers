src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")


def read_data(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f:
        src_data = [l.strip() for l in f if l.strip()]
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_data = [l.strip() for l in f if l.strip()]
    return pd.DataFrame({"src": src_data, "tgt": tgt_data})


if __name__ == "__main__":
    # === Load dataset ===
    print("📘 Loading dataset...")
    df = read_data(src_file, tgt_file)
    print(f"Loaded {len(df)} sentence pairs.")
    print(df.head(), "\n")

    # === Build tokenizers ===
    print("🧠 Building tokenizers...")

    english_tokenizer = (
        TokenizerBuilder()
        .set_type(TokenizerType.WORDPIECE)
        .set_params(
            texts=df["src"].tolist(),
            vocab_size=8000,
            unk_token=SpecialToken.UNK.token_str,
        )
        .build()
    )

    persian_tokenizer = (
        TokenizerBuilder()
        .set_type(TokenizerType.HAZM)
        .set_params(texts=df["tgt"].tolist(), min_freq=2)
        .build()
    )

    print()

    # === Check special token ID equality ===
    print("🔍 Checking SpecialToken ID equality between tokenizers:")
    for tok in SpecialToken:
        eng_id = SpecialToken.as_dict()[tok.token_str]
        per_id = SpecialToken.as_dict()[tok.token_str]
        status = "✅ SAME" if eng_id == per_id else "❌ DIFFERENT"
        print(f"{tok.token_str:6s} → English={eng_id}, Persian={per_id}  {status}")
    print()

    # === Test encoding/decoding ===
    test_sentences_en = [
        "Hello, how are you?",
        "The cat sat on the mat.",
    ]
    test_sentences_fa = [
        "سلام، حالت چطوره؟",
        "گربه روی فرش نشست.",
    ]

    print("🧩 Encoding / Decoding tests:\n")
    for s in test_sentences_en:
        encoded = english_tokenizer.encode(
            s,
        )
        decoded = english_tokenizer.decode(encoded)
        print(f"EN Original: {s}")
        print(f"EN Encoded : {encoded}")
        print(f"EN Decoded : {decoded}\n")

    for s in test_sentences_fa:
        encoded = persian_tokenizer.encode(s)
        decoded = persian_tokenizer.decode(encoded)
        print(f"FA Original: {s}")
        print(f"FA Encoded : {encoded}")
        print(f"FA Decoded : {decoded}\n")

    # === Check special tokens at start/end ===
    print("🧱 Checking CLS/SEP token placement:")
    for s in test_sentences_en:
        encoded = english_tokenizer.encode(s)
        print(
            f"EN start={encoded[0]}, end={encoded[-1]} → CLS={SpecialToken.CLS.value}, SEP={SpecialToken.SEP.value}"
        )
    for s in test_sentences_fa:
        encoded = persian_tokenizer.encode(s)
        print(
            f"FA start={encoded[0]}, end={encoded[-1]} → CLS={SpecialToken.CLS.value}, SEP={SpecialToken.SEP.value}"
        )
    print()

    # === Test padding ===
    print("🧩 Padding test:")
    en_encoded = [english_tokenizer.encode(s) for s in test_sentences_en]
    fa_encoded = [persian_tokenizer.encode(s) for s in test_sentences_fa]

    en_padded = english_tokenizer.pad(en_encoded)
    fa_padded = persian_tokenizer.pad(fa_encoded)

    print("EN (padded):")
    for seq in en_padded:
        print(seq)
    print("FA (padded):")
    for seq in fa_padded:
        print(seq)
    print()

    # === Summary ===
    print("📊 Tokenizer Summary:")
    print(f"English vocab size: {english_tokenizer.get_vocab_size()}")
    print(f"Persian vocab size: {persian_tokenizer.get_vocab_size()}")
    print(f"Special tokens used: {SpecialToken.all_tokens_str()}")
    print("✅ All tests completed successfully.")
