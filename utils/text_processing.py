import re
import contractions
import emoji
import string
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure resources are downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# 1. Typo Map Regex
TYPO_MAP = {
    "did'nt": "did not",
    "didnt": "did not",
    "does'nt": "does not",
    "doesnt": "does not",
    "is'nt": "is not",
    "isnt": "is not",
    "was'nt": "was not",
    "wasnt": "was not",
    "ai'nt": "ain't",
    "aint": "ain't",
    "ca'nt": "cannot",
    "cant": "cannot",
    "wo'nt": "will not",
    "wont": "will not",
    "do'nt": "do not",
    "dont": "do not",
    "should'nt": "should not",
    "shouldnt": "should not",
    "could'nt": "could not",
    "couldnt": "could not",
    "would'nt": "would not",
    "wouldnt": "would not",
}
TYPO_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in TYPO_MAP.keys()) + r")\b"
)

# 2. Emoticon Regex
EMOTICON_MAP = {
    ":)": " smile ",
    ":-)": " smile ",
    ";)": " wink ",
    ":(": " sad ",
    ":-(": " sad ",
    ":D": " laugh ",
    "xD": " laugh ",
    "<3": " love ",
}
# Sort by length desc to match ":-)" before ":)"
EMOTICON_PATTERN = re.compile(
    r"("
    + "|".join(re.escape(k) for k in sorted(EMOTICON_MAP, key=len, reverse=True))
    + r")"
)

# 3. Other Regexes
ELONGATED_WORD_PATTERN = re.compile(r"(.)\1{2,}")
HTML_TAG_PATTERN = re.compile(r"<.*?>")

# 4. Sets for O(1) lookups
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "barely", "hardly", "n't"}
RESET_PUNCTUATION = {".", "?", "!", ";", ","}
SKIP_TOKENS = {"``", "''", "--"}
ARTIFACTS = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Fast Regex Replacements
    if "'" in text:
        text = TYPO_PATTERN.sub(lambda x: TYPO_MAP[x.group()], text)

    text = HTML_TAG_PATTERN.sub(" ", text)
    text = EMOTICON_PATTERN.sub(lambda x: EMOTICON_MAP[x.group()], text)
    text = ELONGATED_WORD_PATTERN.sub(r"\1", text)

    # 3. Expand Contractions
    text = contractions.fix(text)

    # 4. Emojis
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 5. Tokenization
    tokens = word_tokenize(text)

    clean_tokens = []
    negation_active = False
    append = clean_tokens.append

    for token in tokens:
        if token in RESET_PUNCTUATION:
            negation_active = False
            continue

        if token in NEGATION_WORDS:
            negation_active = True
            append(token)
            continue

        if token in string.punctuation or token in SKIP_TOKENS:
            continue

        # Strip leading/trailing quotes immediately.
        token = token.replace("_NOT", "").strip(string.punctuation)

        # If the token was just a quote (e.g. ' ), it becomes empty. Skip it.
        if not token:
            continue

        # Now 'token' is clean so we use it as base_word
        base_word = token

        # Filter checks
        if base_word in STOP_WORDS:
            continue
        if base_word.isdigit():
            continue
        if base_word in ARTIFACTS:
            continue
        if len(base_word) < 2 and base_word not in ("a", "i"):
            continue

        # Append the CLEAN token (with tag if needed)
        if negation_active:
            append(token + "_NOT")
        else:
            append(token)

    return " ".join(clean_tokens)


def load_spacy_model():
    """
    Loads the spacy model with unnecessary components disabled for speed.

    For POS tagging and Lemmatization, we only need 'tagger' and 'attribute_ruler'.
    We disable 'ner' (Named Entity Recognition) and 'parser' (Dependency Parsing)
    as they are computationally expensive and not needed here.
    """
    try:
        # Disable parser and ner for a 5x-10x speedup
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("SpaCy model loaded successfully (parser and NER disabled).")
        return nlp
    except OSError:
        print("Error: Model not found. Run: python -m spacy download en_core_web_sm")
        return None


def process_text_data(df: pd.DataFrame, text_col: str, class_col: str) -> pd.DataFrame:
    """
    Processes text using Spacy's nlp.pipe for high-performance batch processing.

    Performs:
    1. POS Counting
    2. Lemmatization

    Args:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Column name containing text.
        class_col (str): Column name containing sentiment class.

    Returns:
        pd.DataFrame: DataFrame with new columns 'pos_counts' and 'lemmas'.
    """
    nlp = load_spacy_model()
    if not nlp:
        return df

    # Pre-allocate lists
    pos_counts_list = []
    lemmas_list = []

    print(f"Processing {len(df)} documents. This uses nlp.pipe for speed...")

    # nlp.pipe processes texts as a stream/batch, much faster than loop
    # batch_size=1000 is a good sweet spot for memory/speed
    docs = nlp.pipe(df[text_col].astype(str), batch_size=1000, n_process=1)

    for doc in tqdm(docs, total=len(df), desc="NLP Pipeline"):
        # 1. POS Counting (Coarse-grained: NOUN, VERB, ADJ)
        # We filter out PUNCT, SPACE, etc., to keep the plot clean
        relevant_pos = [
            token.pos_
            for token in doc
            if token.pos_ not in {"PUNCT", "SPACE", "X", "SYM"}
        ]
        pos_counts_list.append(Counter(relevant_pos))

        # 2. Lemmatization
        # We extract the lemma only if it's not a stop word or punctuation / number
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
        lemmas_list.append(" ".join(lemmas))

    # Assign results back to DataFrame
    df["pos_counts"] = pos_counts_list
    df["lemmas"] = lemmas_list

    return df


def plot_pos_distribution(df: pd.DataFrame, class_col: str):
    """
    Aggregates POS counts by class and plots the distribution.
    """
    # Aggregate counts
    class_pos_data = []

    classes = df[class_col].unique()

    for cls in classes:
        # Sum all Counters for this class
        subset = df[df[class_col] == cls]["pos_counts"]
        total_counter = sum(subset, Counter())

        # Calculate total tokens for normalization (optional, but good for comparison)
        total_tokens = sum(total_counter.values())

        for pos, count in total_counter.most_common():
            class_pos_data.append(
                {
                    "Class": cls,
                    "POS": pos,
                    "Count": count,
                    "Frequency": count / total_tokens if total_tokens > 0 else 0,
                }
            )

    pos_df = pd.DataFrame(class_pos_data)

    # Plotting
    plt.figure(figsize=(14, 6))

    # We plot the raw count, but you could switch y to 'Frequency' for normalized view
    chart = sns.barplot(data=pos_df, x="POS", y="Count", hue="Class", palette="viridis")

    plt.title("Distribution of Parts of Speech (POS) by Sentiment Class", fontsize=16)
    plt.ylabel("Total Occurrences")
    plt.xlabel("Part of Speech Tag")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.show()


def plot_word_clouds(df: pd.DataFrame, class_col: str):
    """
    Generates and plots word clouds for each class using lemmatized text.
    """
    classes = df[class_col].unique()

    fig, axes = plt.subplots(1, len(classes), figsize=(20, 10))
    if len(classes) == 1:
        axes = [axes]  # Handle single class case

    for ax, cls in zip(axes, classes):
        # Join all lemmas for this class into one massive string
        text_corpus = " ".join(df[df[class_col] == cls]["lemmas"])

        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="black",
            min_font_size=10,
            colormap="viridis" if cls == "positive" else "magma",
        ).generate(text_corpus)

        ax.imshow(wordcloud)
        ax.axis("off")
        ax.set_title(f"Word Cloud: {cls.capitalize()} Reviews", fontsize=16)

    plt.tight_layout()
    plt.show()
