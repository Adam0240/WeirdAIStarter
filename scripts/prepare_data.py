#allows the script to download the dataset
from datasets import load_dataset
from weird_ai.config import RAW_DATA_DIR, SAMPLE_LYRICS_FILE, PROCESSED_DATA_DIR

"""
Read the downloaded dataset (parquet files) and create lyrics_sample.txt.
"""
def main():
    #creates a list limited to the first 5000 songs in the dataset. 
    lyrics_column = "lyrics"
    limit = 5000
    selected_lyrics = []

    #creates the directory if it does not exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    #loads the raw datasets from the .parquet files
    dataset = load_dataset(
        "parquet",
        data_files=str(RAW_DATA_DIR / "*.parquet"),
        split="train"
    )

    print(dataset)
    print(dataset.column_names)

    #Iterate through the dataset one entry at a time
    for row in dataset:
        lyrics = row.get(lyrics_column)

        #rejects entries without lyrics
        if lyrics is None:
            continue

        #removes extra whitespace
        lyrics = lyrics.strip()

        #Rejects lyrics with less than 100 characters
        if len(lyrics) < 100:
            continue

        #stores selected lyrics
        selected_lyrics.append(lyrics)

        #ends the loop once 5000 limit has been reached
        if len(selected_lyrics) >= limit:
            break

    #combines lyrics into one large text file
    output_text = "\n\n<|song|>\n\n".join(selected_lyrics)

    #writes the result to the text file
    SAMPLE_LYRICS_FILE.write_text(output_text, encoding="utf-8")

    print(f"Songs written: {len(selected_lyrics)}")
    print(f"Output file: {SAMPLE_LYRICS_FILE}")


if __name__ == "__main__":
    main()
