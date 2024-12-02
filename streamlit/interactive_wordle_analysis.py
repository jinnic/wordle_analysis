import streamlit as st
import pandas as pd
import numpy as np
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
def plot_frequency_bar(freq_dict, title, color):
    sorted_freq = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_freq.keys(), sorted_freq.values(), color=color)
    plt.title(title, fontsize=16)
    plt.xlabel("Letters", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
        )
    st.pyplot(plt)


def plot_positional_heatmap(positional_freq_dict, title, cmap):
    heatmap_data = pd.DataFrame(positional_freq_dict).fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, cmap=cmap, annot=True, fmt=".0f", cbar=True, linewidths=0.5
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Letters", fontsize=12)
    st.pyplot(plt)


# Load Data
st.title("Wordle Data Analysis")
wordle_words, five_letter_words = load_data()

# Select Dataset
dataset_option = st.radio("Select Dataset to Explore:", ("Wordle", "English"))

if dataset_option == "Wordle":
    word_list = wordle_words
    dataset_name = "Wordle Dataset"
    color = "skyblue"
else:
    word_list = five_letter_words
    dataset_name = "English Dataset"
    color = "lightgreen"

# Frequency Analysis
st.header(f"Letter Frequency in {dataset_name}")
letter_freq = calculate_letter_frequency(word_list)
plot_frequency_bar(letter_freq, f"Letter Frequency in {dataset_name}", color)

# Positional Analysis
st.header(f"Positional Frequency in {dataset_name}")
positional_freq = calculate_positional_frequency(word_list)
plot_positional_heatmap(
    positional_freq, f"Positional Frequency in {dataset_name}", cmap="coolwarm"
)

# Deviation Analysis
st.header("Deviation Analysis")
if dataset_option == "Wordle":
    english_letter_freq = calculate_letter_frequency(five_letter_words)
    wordle_prob = calculate_probabilities(letter_freq, sum(letter_freq.values()))
    english_prob = calculate_probabilities(
        english_letter_freq, sum(english_letter_freq.values())
    )
    deviation = (
        sum(
            abs(wordle_prob.get(k, 0) - english_prob.get(k, 0))
            for k in set(wordle_prob.keys()).union(english_prob.keys())
        )
        / 2
    )
    st.write(
        f"The Total Variation Distance (TVD) between Wordle and English datasets is: **{deviation:.4f}**"
    )
