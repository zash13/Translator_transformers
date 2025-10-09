import os
from transformers import AutoTokenizer
import re

#


def analyze_persian_tokenizer_vocabulary(tokenizer, top_n=2000):
    """Analyze and print Persian tokenizer vocabulary in detail"""
    print(f"\nPERSIAN TOKENIZER VOCABULARY ANALYSIS (First {top_n} tokens):")
    print("=" * 80)

    vocab = tokenizer.vocab
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print(f"Total vocabulary size: {len(vocab)}")

    # Get first tokens
    tokens_only = [token for token, idx in sorted_vocab[:top_n]]

    # Group by categories for analysis
    persian_tokens = [t for t in tokens_only if re.search(r"[\u0600-\u06FF]", t)]
    english_tokens = [
        t
        for t in tokens_only
        if re.search(r"[a-zA-Z]", t) and not re.search(r"[\u0600-\u06FF]", t)
    ]
    digit_tokens = [t for t in tokens_only if re.search(r"\d", t)]
    punctuation_tokens = [
        t for t in tokens_only if re.search(r"[^\w\s\u0600-\u06FF]", t) and len(t) == 1
    ]
    special_tokens = [t for t in tokens_only if t.startswith("[") and t.endswith("]")]
    subword_tokens = [t for t in tokens_only if t.startswith("##")]
    other_tokens = [
        t
        for t in tokens_only
        if t
        not in persian_tokens
        + english_tokens
        + digit_tokens
        + punctuation_tokens
        + special_tokens
        + subword_tokens
    ]

    print(f"\n📊 VOCABULARY BREAKDOWN:")
    print(f"  Persian tokens: {len(persian_tokens)}")
    print(f"  English tokens: {len(english_tokens)}")
    print(f"  Digit tokens: {len(digit_tokens)}")
    print(f"  Punctuation: {len(punctuation_tokens)}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Subword tokens: {len(subword_tokens)}")
    print(f"  Other tokens: {len(other_tokens)}")

    # Calculate garbage percentage (non-Persian scripts)
    garbage_scripts = 0
    for token in tokens_only:
        if (
            re.search(r"[\u0900-\u097F]", token)  # Devanagari
            or re.search(r"[\u0980-\u09FF]", token)  # Bengali
            or re.search(r"[\u0A00-\u0A7F]", token)  # Gurmukhi
            or re.search(r"[\u0A80-\u0AFF]", token)  # Gujarati
            or re.search(r"[\u0B00-\u0B7F]", token)  # Oriya
            or re.search(r"[\u0B80-\u0BFF]", token)  # Tamil
            or re.search(r"[\u0C00-\u0C7F]", token)  # Telugu
            or re.search(r"[\u0C80-\u0CFF]", token)  # Kannada
            or re.search(r"[\u0D00-\u0D7F]", token)  # Malayalam
            or re.search(r"[\u0E00-\u0E7F]", token)  # Thai
            or re.search(r"[\u0E80-\u0EFF]", token)  # Lao
            or re.search(r"[\u0F00-\u0FFF]", token)  # Tibetan
            or re.search(r"[\u1000-\u109F]", token)  # Myanmar
            or re.search(r"[\u1780-\u17FF]", token)  # Khmer
            or re.search(r"[\u4E00-\u9FFF]", token)
        ):  # CJK
            garbage_scripts += 1

    print(
        f"  Garbage script tokens: {garbage_scripts} ({garbage_scripts / top_n * 100:.1f}%)"
    )

    # Print ALL first tokens in organized format
    print(f"\n" + "=" * 80)
    print(f"COMPLETE LIST OF FIRST {len(tokens_only)} TOKENS:")
    print("=" * 80)

    # Print in columns of 10 tokens per line
    for i in range(0, len(tokens_only), 10):
        end_idx = min(i + 10, len(tokens_only))
        line_tokens = tokens_only[i:end_idx]

        # Format line with fixed width for alignment
        formatted_line = ""
        for j, token in enumerate(line_tokens):
            # Handle special characters and display
            display_token = token
            if token == "":
                display_token = "''"
            elif token == " ":
                display_token = "' '"
            elif token == "\t":
                display_token = "'\\t'"
            elif token == "\n":
                display_token = "'\\n'"

            # Color code by type
            if token in special_tokens:
                formatted_line += (
                    f"\033[95m{display_token:<12}\033[0m"  # Magenta for special
                )
            elif token in persian_tokens:
                formatted_line += (
                    f"\033[92m{display_token:<12}\033[0m"  # Green for Persian
                )
            elif token in english_tokens:
                formatted_line += (
                    f"\033[94m{display_token:<12}\033[0m"  # Blue for English
                )
            elif token in subword_tokens:
                formatted_line += (
                    f"\033[93m{display_token:<12}\033[0m"  # Yellow for subword
                )
            elif token in punctuation_tokens:
                formatted_line += (
                    f"\033[91m{display_token:<12}\033[0m"  # Red for punctuation
                )
            else:
                # Check if it's garbage script
                if (
                    re.search(r"[\u0900-\u097F]", token)
                    or re.search(r"[\u0980-\u09FF]", token)
                    or re.search(r"[\u0A00-\u0A7F]", token)
                    or re.search(r"[\u0A80-\u0AFF]", token)
                    or re.search(r"[\u0B00-\u0B7F]", token)
                    or re.search(r"[\u0B80-\u0BFF]", token)
                    or re.search(r"[\u0C00-\u0C7F]", token)
                    or re.search(r"[\u0C80-\u0CFF]", token)
                    or re.search(r"[\u0D00-\u0D7F]", token)
                    or re.search(r"[\u0E00-\u0E7F]", token)
                    or re.search(r"[\u0E80-\u0EFF]", token)
                    or re.search(r"[\u0F00-\u0FFF]", token)
                    or re.search(r"[\u1000-\u109F]", token)
                    or re.search(r"[\u1780-\u17FF]", token)
                    or re.search(r"[\u4E00-\u9FFF]", token)
                ):
                    formatted_line += f"\033[41m{display_token:<12}\033[0m"  # Red background for garbage
                else:
                    formatted_line += f"{display_token:<12}"

        print(f"{i:4d}-{end_idx - 1:4d}: {formatted_line}")

    # Print legend
    print(f"\nLEGEND: ")
    print(f"\033[95mMagenta\033[0m = Special tokens")
    print(f"\033[92mGreen\033[0m = Persian tokens")
    print(f"\033[94mBlue\033[0m = English tokens")
    print(f"\033[93mYellow\033[0m = Subword tokens")
    print(f"\033[91mRed\033[0m = Punctuation")
    print(f"\033[41mRed Background\033[0m = Garbage script tokens")
    print(f"White = Other tokens")


def test_performance_on_persian_paragraph(tokenizer, tokenizer_name):
    """Test tokenizer performance on Persian paragraph"""
    print(f"\n" + "=" * 60)
    print(f"PERFORMANCE TEST: {tokenizer_name}")
    print("=" * 60)

    # Persian paragraph for testing
    persian_paragraph = """
    زبان فارسی که با نام‌های فارسی، پارسی، و فارسی نیز شناخته می‌شود، یکی از زبان‌های هندواروپایی در شاخهٔ زبان‌های ایرانی جنوب غربی است. 
    این زبان در کشورهای ایران، افغانستان، تاجیکستان و ازبکستان به طور رسمی صحبت می‌شود و دارای تاریخی کهن و ادبیاتی غنی است. 
    فارسی نوین که امروزه مورد استفاده قرار می‌گیرد، از فارسی میانه و آن نیز از فارسی باستان نشأت گرفته است. 
    متون ادبی فارسی شامل آثار بزرگی همچون شاهنامه فردوسی، دیوان حافظ، غزلیات سعدی، و مثنوی مولانا می‌باشد که هر یک گنجینه‌ای از فرهنگ و ادب پارسی محسوب می‌شوند.
    زبان فارسی با خط فارسی نوشته می‌شود که adaptationsی از خط عربی است و دارای ۳۲ حرف می‌باشد. 
    این زبان در طول تاریخ تأثیرات زیادی بر زبان‌های منطقه از جمله ترکی، اردو، و هندی داشته است.
    """

    # Clean the paragraph
    persian_paragraph = " ".join(persian_paragraph.split())

    print(f"Test Paragraph ({len(persian_paragraph)} characters):")
    print(f"'{persian_paragraph[:100]}...'")

    # Tokenize tokens = tokenizer.tokenize(persian_paragraph)
    print(f"\nTokenization Results:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")

    # Calculate compression ratio
    char_count = len(persian_paragraph)
    token_count = len(tokens)
    compression_ratio = char_count / token_count if token_count > 0 else 0

    print(f"\n📈 Performance Metrics:")
    print(f"  Characters: {char_count}")
    print(f"  Tokens: {token_count}")
    print(f"  Compression ratio: {compression_ratio:.2f} chars/token")

    # Analyze token quality
    persian_token_count = sum(1 for t in tokens if re.search(r"[\u0600-\u06FF]", t))
    garbage_token_count = sum(
        1
        for t in tokens
        if any(
            [
                re.search(r"[\u0900-\u097F]", t),
                re.search(r"[\u0980-\u09FF]", t),
                re.search(r"[\u0A00-\u0A7F]", t),
                re.search(r"[\u0A80-\u0AFF]", t),
                re.search(r"[\u0B00-\u0B7F]", t),
                re.search(r"[\u0B80-\u0BFF]", t),
                re.search(r"[\u0C00-\u0C7F]", t),
                re.search(r"[\u0C80-\u0CFF]", t),
                re.search(r"[\u0D00-\u0D7F]", t),
                re.search(r"[\u0E00-\u0E7F]", t),
                re.search(r"[\u0E80-\u0EFF]", t),
                re.search(r"[\u0F00-\u0FFF]", t),
                re.search(r"[\u1000-\u109F]", t),
                re.search(r"[\u1780-\u17FF]", t),
                re.search(r"[\u4E00-\u9FFF]", t),
            ]
        )
    )

    print(
        f"  Persian tokens: {persian_token_count} ({persian_token_count / token_count * 100:.1f}%)"
    )
    print(
        f"  Garbage tokens: {garbage_token_count} ({garbage_token_count / token_count * 100:.1f}%)"
    )

    return tokens


def main2_persian_test():
    # Path to Persian file
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("Loading Persian TEP data...")
    try:
        with open(tgt_file, "r", encoding="utf-8") as f:
            persian_sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {tgt_file}")
        # Create realistic Persian sample data
        persian_sentences = [
            "من دوست دارم برنامه نویسی کنم",
            "این یک متن فارسی است",
            "هوا امروز بسیار خوب است",
            "کتاب را روی میز گذاشتم",
            "دانشگاه تهران یکی از بهترین دانشگاه‌های ایران است",
            "در جهان امروز، فناوری اطلاعات نقش مهمی دارد",
            "زبان فارسی دارای تاریخ کهنی است",
            "ایران کشوری با تمدن باستانی است",
            "علم و دانش اهمیت زیادی در زندگی انسان دارد",
            "خواندن کتاب باعث افزایش اطلاعات می‌شود",
        ] * 20000

    print(f"Loaded {len(persian_sentences)} Persian sentences.")

    # Limit for testing
    sample_size = min(200000, len(persian_sentences))
    persian_sentences = persian_sentences[:sample_size]

    print("\n" + "=" * 60)
    print("PERSIAN BPE TOKENIZER FOR PERSIAN TRANSLATION MODEL")
    print("=" * 60)

    # Load Persian BPE Tokenizer
    print("\nLoading PersianBPE tokenizer...")
    try:
        # Try to load from Hugging Face hub
        persian_tokenizer = AutoTokenizer.from_pretrained(
            "mshojaei77/PersianBPETokenizer"
        )

        # Test basic functionality
        test_sentence = "من دوست دارم برنامه نویسی کنم و این کار را بسیار دوست دارم"
        tokens = persian_tokenizer.tokenize(test_sentence)

        print(f"\n🧪 BASIC TEST:")
        print(f"Sentence: '{test_sentence}'")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Vocabulary size: {len(persian_tokenizer)}")

        # Analyze vocabulary - PRINT FIRST 2000 TOKENS
        analyze_persian_tokenizer_vocabulary(persian_tokenizer, 4000)

        # Test performance on Persian paragraph
        test_performance_on_persian_paragraph(persian_tokenizer, "PersianBPE Tokenizer")

        # Test with complex Persian words
        print("\n" + "=" * 60)
        print("COMPLEX PERSIAN WORDS TEST:")
        print("=" * 60)

        complex_sentences = [
            "الگوریتم‌های یادگیری عمیق در هوش مصنوعی استفاده می‌شوند",
            "بیوانفورماتیک شاخه‌ای میان‌رشته‌ای است",
            "نانوتکنولوژی تحول بزرگی در علم ایجاد کرده است",
            "پارسی‌گویی و فارسی‌نویسی اهمیت زیادی دارد",
            "کتاب‌خانه‌های دیجیتال امروزه بسیار پرکاربرد هستند",
        ]

        for i, sentence in enumerate(complex_sentences, 1):
            tokens = persian_tokenizer.tokenize(sentence)
            print(f"\n{i}. '{sentence}'")
            print(f"   Tokens: {tokens}")
            print(f"   Count: {len(tokens)} tokens")

    except Exception as e:
        print(f"Error with PersianBPE tokenizer: {e}")
        print("Trying to use local tokenizer or fallback...")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main2_persian_test()
