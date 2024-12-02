import os
import requests
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Section 1: Data Fetching
def save_five_letter_words_to_file(file_path="five_letter_words.txt"):
    """
    Saves NLTK five-letter words to a text file. If the file already exists,
    skips NLTK processing and reads from the file.

    Args:
    file_path (str): The path to save or read the five-letter words file.

    Returns:
    list: A list of five-letter words.
    """
    if os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Loading words from the file...")
        with open(file_path, "r") as file:
            words = file.read().splitlines()
    else:
        print(
            f"File '{file_path}' does not exist. Fetching five-letter words from NLTK..."
        )
        import nltk

        nltk.download("words")
        from nltk.corpus import words as nltk_words

        # Fetch and filter five-letter words
        words = [word.lower() for word in nltk_words.words() if len(word) == 5]

        # Save to file
        with open(file_path, "w") as file:
            file.write("\n".join(words))
        print(f"Five-letter words saved to '{file_path}'.")

    return words


def fetch_wordle_answers(url):
    """Fetches Wordle answers from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        section_header = soup.find(
            "h3", id="section-past-wordle-answers-alphabetical-list"
        )
        if section_header:
            answers_paragraph = section_header.find_next("p")
            if answers_paragraph:
                return answers_paragraph.get_text(strip=True)
            else:
                raise ValueError("Could not find the <p> tag after the <h3> header.")
        else:
            raise ValueError("Could not find the specified <h3> header.")
    else:
        raise ConnectionError(
            f"Failed to fetch data. Status code: {response.status_code}"
        )


# Section 2: File Operations
def save_to_file(content, file_path):
    """Saves content to a file."""
    with open(file_path, "w") as file:
        file.write(content)


def read_from_file(file_path):
    """Reads content from a file."""
    with open(file_path, "r") as file:
        return file.read()


# Section 3: Data Processing
def split_words_to_list(word_string, delimiter="|"):
    """Splits a string of words by a delimiter into a list."""
    return [word.strip().lower() for word in word_string.split(delimiter)]


def calculate_letter_frequency(word_list):
    """Calculates the frequency of each letter in a list of words."""
    all_letters = "".join(word_list)
    return Counter(all_letters)


def calculate_positional_frequency(word_list):
    """Calculates the frequency of each letter by position in the words."""
    position_counts = {}
    word_length = len(word_list[0])  # Assumes all words have the same length
    for position in range(word_length):
        letters_at_position = [word[position] for word in word_list]
        position_counts[position + 1] = Counter(letters_at_position)
    return position_counts


def get_top_letters(freq_dict, top_n=10):
    return pd.DataFrame(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n],
        columns=["Letter", "Frequency"],
    )


# Section 4: Statistics and Probabilities
def calculate_statistics(freq_dict):
    """Calculates statistics for a frequency dictionary."""
    frequencies = np.array(list(freq_dict.values()))
    mean = np.mean(frequencies)
    std_dev = np.std(frequencies)
    normalized = {k: (v - mean) / std_dev for k, v in freq_dict.items()}
    return {"mean": mean, "std_dev": std_dev, "normalized": normalized}


def calculate_probabilities(freq_dict, total_count):
    """Calculates probabilities for each letter."""
    return {k: v / total_count for k, v in freq_dict.items()}


def calculate_positional_probabilities(positional_freq_dict):
    """Calculates the probability of each letter appearing at each position."""
    positional_probs = {}
    for position, freq_dict in positional_freq_dict.items():
        total_letters_at_position = sum(freq_dict.values())
        positional_probs[position] = {
            letter: count / total_letters_at_position
            for letter, count in freq_dict.items()
        }
    return positional_probs


# Section 5: Visualize Frequencies
def plot_frequency_bar(freq_dict, title, color="skyblue"):
    """
    Plots a bar chart for letter frequencies with annotations.

    Args:
    freq_dict (dict): Frequency dictionary of letters.
    title (str): Title of the plot.
    color (str): Color of the bars.
    """
    sorted_freq = dict(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    )  # Sort by frequency
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_freq.keys(), sorted_freq.values(), color=color)
    plt.title(title, fontsize=16)
    plt.xlabel("Letters", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 10,
            f"{int(height)}",
            ha="center",
            fontsize=10,
        )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    # def plot_frequency_bar(freq_dict, title):
    # """Plots a bar chart for letter frequencies."""
    # plt.figure(figsize=(10, 6))
    # plt.bar(freq_dict.keys(), freq_dict.values(), color="skyblue")
    # plt.title(title)
    # plt.xlabel("Letters")
    # plt.ylabel("Frequency")
    # plt.show()


def plot_positional_heatmap(positional_freq_dict, title, cmap="coolwarm"):
    """
    Plots a heatmap for positional letter frequencies with normalized data and alphabet on Y-axis.

    Args:
    positional_freq_dict (dict): Positional frequency dictionary.
    title (str): Title of the heatmap.
    cmap (str): Colormap for the heatmap.
    """
    heatmap_data = pd.DataFrame(positional_freq_dict).fillna(0)
    heatmap_data = heatmap_data.div(
        heatmap_data.sum(axis=0), axis=1
    )  # Normalize by column (position)
    heatmap_data.index = list("abcdefghijklmnopqrstuvwxyz")[
        : heatmap_data.shape[0]
    ]  # Set Y-axis as alphabet

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, cmap=cmap, annot=True, fmt=".2f", cbar=True, linewidths=0.5
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Letters", fontsize=12)
    plt.tight_layout()
    plt.show()

    # def plot_positional_heatmap(positional_freq_dict, title):
    # """Plots a heatmap for positional letter frequencies."""
    # heatmap_data = pd.DataFrame(positional_freq_dict).fillna(0)
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".0f")
    # plt.title(title)
    # plt.xlabel("Position")
    # plt.ylabel("Letters")
    # plt.show()


def visualize_data(
    wordle_freq, wordle_positional_freq, english_freq, english_positional_freq
):
    """
    Visualizes letter and positional frequencies in a more effective way.

    Args:
    wordle_freq (dict): Letter frequencies for Wordle answers.
    wordle_positional_freq (dict): Positional frequencies for Wordle answers.
    english_freq (dict): Letter frequencies for English five-letter words.
    english_positional_freq (dict): Positional frequencies for English five-letter words.
    """
    # Wordle Letter Frequency
    print("## plot frequency bar - wordle")
    plot_frequency_bar(wordle_freq, "Wordle Letter Frequency", color="skyblue")

    # Wordle Positional Frequency Heatmap
    print("## plot positional heatmap - wordle")
    plot_positional_heatmap(
        wordle_positional_freq, "Wordle Positional Frequency", cmap="Blues"
    )

    # English Letter Frequency
    print("## plot frequency bar - english")
    plot_frequency_bar(english_freq, "English Letter Frequency", color="lightgreen")

    # English Positional Frequency Heatmap
    print("## plot positional heatmap - english")
    plot_positional_heatmap(
        english_positional_freq, "English Positional Frequency", cmap="Greens"
    )


# def visualize_data(
#     wordle_freq, wordle_positional_freq, english_freq, english_positional_freq
# ):
#     """
#     Visualizes letter and positional frequencies in continuous Matplotlib figures.

#     Args:
#     wordle_freq (dict): Letter frequencies for Wordle answers.
#     wordle_positional_freq (dict): Positional frequencies for Wordle answers.
#     english_freq (dict): Letter frequencies for English five-letter words.
#     english_positional_freq (dict): Positional frequencies for English five-letter words.
#     """
#     fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 grid of subplots

#     # Plot 1: Wordle Letter Frequency
#     axs[0, 0].bar(wordle_freq.keys(), wordle_freq.values(), color="skyblue")
#     axs[0, 0].set_title("Wordle Letter Frequency")
#     axs[0, 0].set_xlabel("Letters")
#     axs[0, 0].set_ylabel("Frequency")

#     # Plot 2: Wordle Positional Frequency Heatmap
#     heatmap_data_wordle = pd.DataFrame(wordle_positional_freq).fillna(0)
#     sns.heatmap(heatmap_data_wordle, cmap="Blues", annot=True, fmt=".0f", ax=axs[0, 1])
#     axs[0, 1].set_title("Wordle Positional Frequency")

#     # Plot 3: English Letter Frequency
#     axs[1, 0].bar(english_freq.keys(), english_freq.values(), color="lightgreen")
#     axs[1, 0].set_title("English Five-Letter Words Letter Frequency")
#     axs[1, 0].set_xlabel("Letters")
#     axs[1, 0].set_ylabel("Frequency")

#     # Plot 4: English Positional Frequency Heatmap
#     heatmap_data_english = pd.DataFrame(english_positional_freq).fillna(0)
#     sns.heatmap(
#         heatmap_data_english, cmap="Greens", annot=True, fmt=".0f", ax=axs[1, 1]
#     )
#     axs[1, 1].set_title("English Positional Frequency")

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()


# Section 6: Analyze Deviation


def total_variation_distance(prob_dist1, prob_dist2):
    """
    Computes the total variation distance (TVD) between two probability distributions.

    Args:
    prob_dist1 (dict): First probability distribution.
    prob_dist2 (dict): Second probability distribution.

    Returns:
    float: Total variation distance.
    """
    keys = set(prob_dist1.keys()).union(prob_dist2.keys())
    return sum(abs(prob_dist1.get(k, 0) - prob_dist2.get(k, 0)) for k in keys) / 2


def analyze_deviation(wordle_list, five_letter_words):
    """
    Analyzes deviation between Wordle word distributions and English five-letter word distributions.

    Args:
    wordle_list (list): List of Wordle answers.
    five_letter_words (list): List of English five-letter words.

    Returns:
    None
    """
    # Calculate frequencies
    wordle_freq = calculate_letter_frequency(wordle_list)
    english_freq = calculate_letter_frequency(five_letter_words)

    # Normalize to probabilities
    wordle_probs = calculate_probabilities(wordle_freq, sum(wordle_freq.values()))
    english_probs = calculate_probabilities(english_freq, sum(english_freq.values()))

    # Calculate total variation distance
    tvd = total_variation_distance(wordle_probs, english_probs)

    print("\n--- Analysis of Deviation ---")
    print(f"Total Variation Distance: {tvd:.4f}")
    print("\nTop Wordle Letter Probabilities:")
    for k, v in sorted(wordle_probs.items(), key=lambda item: item[1], reverse=True)[
        :10
    ]:
        print(f"{k}: {v:.4f}")
    print("\nTop English Letter Probabilities:")
    for k, v in sorted(english_probs.items(), key=lambda item: item[1], reverse=True)[
        :10
    ]:
        print(f"{k}: {v:.4f}")


# Section 7: Visualize Comparison
def plot_comparison(wordle_probs, english_probs):
    """
    Plots a comparison of Wordle and English letter probability distributions.

    Args:
    wordle_probs (dict): Probability distribution for Wordle answers.
    english_probs (dict): Probability distribution for English words.

    Returns:
    None
    """
    import matplotlib.pyplot as plt

    labels = sorted(set(wordle_probs.keys()).union(english_probs.keys()))
    wordle_values = [wordle_probs.get(k, 0) for k in labels]
    english_values = [english_probs.get(k, 0) for k in labels]

    x = range(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, wordle_values, width=0.4, label="Wordle", align="center", alpha=0.7)
    plt.bar(x, english_values, width=0.4, label="English", align="edge", alpha=0.7)
    plt.xticks(x, labels)
    plt.title("Comparison of Letter Distributions")
    plt.xlabel("Letters")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


# Main Script
def main():
    """
    Main function to orchestrate the data fetching, processing, analysis, and visualization.
    """
    wordle_url = "https://www.techradar.com/news/past-wordle-answers"
    file_path = "wordle.txt"

    try:
        # Step 1: Fetch and Save Wordle Answers
        print("Fetching Wordle answers...")
        wordle_answers = fetch_wordle_answers(wordle_url)
        save_to_file(wordle_answers, file_path)
        print(f"Wordle answers saved to {file_path}.")

        # Step 2: Read and Process Wordle Answers
        print("Processing Wordle answers...")
        wordle_data = read_from_file(file_path)
        wordle_list = split_words_to_list(wordle_data)

        # Step 3: Fetch English Five-Letter Words
        print("Fetching English five-letter words...")
        five_letter_words = save_five_letter_words_to_file()

        # Step 4: Calculate Frequencies
        print("Calculating frequencies...")
        wordle_freq = calculate_letter_frequency(wordle_list)
        wordle_positional_freq = calculate_positional_frequency(wordle_list)
        english_freq = calculate_letter_frequency(five_letter_words)
        english_positional_freq = calculate_positional_frequency(five_letter_words)

        # Step 4-1: Get Top Letters
        top_10_wordle = get_top_letters(wordle_freq)
        top_10_five_letter = get_top_letters(english_freq)

        # Add relative frequency (percentage) for better comparison
        total_wordle = sum(wordle_freq.values())
        total_five_letter = sum(english_freq.values())

        top_10_wordle["Percentage"] = (
            top_10_wordle["Frequency"] / total_wordle * 100
        ).round(2)
        top_10_five_letter["Percentage"] = (
            top_10_five_letter["Frequency"] / total_five_letter * 100
        ).round(2)

        # Step 5: Visualize Frequencies
        print("Visualizing data...")
        visualize_data(
            wordle_freq, wordle_positional_freq, english_freq, english_positional_freq
        )

        # Step 6: Analyze Deviation
        print("Analyzing deviation from uniform sampling...")
        analyze_deviation(wordle_list, five_letter_words)

        # Step 7: Visualize Comparison
        print("Visualizing comparison...")
        wordle_probs = calculate_probabilities(
            calculate_letter_frequency(wordle_list), sum(wordle_freq.values())
        )
        english_probs = calculate_probabilities(
            calculate_letter_frequency(five_letter_words), sum(english_freq.values())
        )
        plot_comparison(wordle_probs, english_probs)

        print("Process completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


# Entry Point
if __name__ == "__main__":
    main()
