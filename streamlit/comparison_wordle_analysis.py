import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Load Data
@st.cache_data
def load_data():
    with open("wordle.txt", "r") as file:
        wordle_words = [word.strip().lower() for word in file.read().split("|")]
    with open("five_letter_words.txt", "r") as file:
        five_letter_words = [word.strip().lower() for word in file.readlines()]
    return wordle_words, five_letter_words


# Calculate Frequencies
def calculate_letter_frequency(word_list):
    all_letters = "".join(word_list)
    return Counter(all_letters)


def calculate_positional_frequency(word_list):
    position_counts = {}
    word_length = len(word_list[0])  # Assumes all words have the same length
    for position in range(word_length):
        letters_at_position = [word[position] for word in word_list]
        position_counts[position + 1] = Counter(letters_at_position)
    return position_counts


def calculate_probabilities(freq_dict, total_count):
    return {k: v / total_count for k, v in freq_dict.items()}


# Visualization Functions
def plot_frequency_bar(freq_dict1, freq_dict2, labels, title):
    """
    Plots side-by-side bar charts comparing two datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    freq1 = [freq_dict1.get(k, 0) for k in labels]
    freq2 = [freq_dict2.get(k, 0) for k in labels]

    bar_width = 0.4
    ax.bar(x, freq1, width=bar_width, label="Wordle", color="skyblue", align="center")
    ax.bar(
        [p + bar_width for p in x],
        freq2,
        width=bar_width,
        label="English",
        color="lightgreen",
        align="center",
    )

    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Letters", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    st.pyplot(fig)


def plot_positional_heatmap(positional_freq1, positional_freq2, title1, title2):
    """
    Plots side-by-side heatmaps comparing positional frequencies of two datasets.
    """
    heatmap1 = pd.DataFrame(positional_freq1).fillna(0)
    heatmap2 = pd.DataFrame(positional_freq2).fillna(0)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(heatmap1, cmap="Blues", annot=True, fmt=".0f", cbar=True, ax=axs[0])
    axs[0].set_title(title1, fontsize=14)
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Letters")

    sns.heatmap(heatmap2, cmap="Greens", annot=True, fmt=".0f", cbar=True, ax=axs[1])
    axs[1].set_title(title2, fontsize=14)
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Letters")

    plt.tight_layout()
    st.pyplot(fig)


# Load Data
st.title("Comparison of Wordle and English Datasets")
wordle_words, five_letter_words = load_data()

# Frequency Analysis
st.header("Letter Frequency Comparison")
wordle_freq = calculate_letter_frequency(wordle_words)
english_freq = calculate_letter_frequency(five_letter_words)

letters = sorted(set(wordle_freq.keys()).union(english_freq.keys()))
plot_frequency_bar(wordle_freq, english_freq, letters, "Letter Frequency Comparison")

# Positional Analysis
st.header("Positional Frequency Comparison")
wordle_positional_freq = calculate_positional_frequency(wordle_words)
english_positional_freq = calculate_positional_frequency(five_letter_words)
plot_positional_heatmap(
    wordle_positional_freq,
    english_positional_freq,
    "Wordle Positional Frequency",
    "English Positional Frequency",
)

# Deviation Analysis
st.header("Deviation Analysis")
wordle_probs = calculate_probabilities(wordle_freq, sum(wordle_freq.values()))
english_probs = calculate_probabilities(english_freq, sum(english_freq.values()))

tvd = sum(abs(wordle_probs.get(k, 0) - english_probs.get(k, 0)) for k in letters) / 2
st.write(
    f"The Total Variation Distance (TVD) between Wordle and English datasets is: **{tvd:.4f}**"
)
