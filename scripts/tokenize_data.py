#Imports lyric and token files from the weird_ai_configuration
from weird_ai.config import SAMPLE_LYRICS_FILE, PROCESSED_DATA_DIR, TOKENS_FILE
from weird_ai.tokenizer import SimpleCharacterTokenizer

"""
Convert text into tokens
"""
def main():
    #Checks if the processed_data folder exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    #reads the lyric sample
    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")

    #creates the tokenizer. this looks at the entire lyric sample and builds its vocab based on the unique characters in the text. assigns each unique character a number
    tokenizer = SimpleCharacterTokenizer(text)

    #encodes the text
    tokens = tokenizer.encode(text)

    #saves the tokens to a file
    TOKENS_FILE.write_text(
        " ".join(str(token) for token in tokens),
        encoding="utf-8"
    )

    print(f"Characters in corpus: {len(text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Tokens written to: {TOKENS_FILE}")

    #decides the first 500 tokens
    decoded_preview = tokenizer.decode(tokens[:500])

    print("\nDecoded preview:")
    print(decoded_preview)


if __name__ == "__main__":
    main()