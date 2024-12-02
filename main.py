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


# Section 5: Visualization and Analysis
def plot_frequency_bar(freq_dict, title):
    """Plots a bar chart for letter frequencies."""
    plt.figure(figsize=(10, 6))
    plt.bar(freq_dict.keys(), freq_dict.values(), color="skyblue")
    plt.title(title)
    plt.xlabel("Letters")
    plt.ylabel("Frequency")
    plt.show()


def plot_positional_heatmap(positional_freq_dict, title):
    """Plots a heatmap for positional letter frequencies."""
    heatmap_data = pd.DataFrame(positional_freq_dict).fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".0f")
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Letters")
    plt.show()


def visualize_data(
    wordle_freq, wordle_positional_freq, english_freq, english_positional_freq
):
    """
    Visualizes letter and positional frequencies in continuous Matplotlib figures.

    Args:
    wordle_freq (dict): Letter frequencies for Wordle answers.
    wordle_positional_freq (dict): Positional frequencies for Wordle answers.
    english_freq (dict): Letter frequencies for English five-letter words.
    english_positional_freq (dict): Positional frequencies for English five-letter words.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 grid of subplots

    # Plot 1: Wordle Letter Frequency
    axs[0, 0].bar(wordle_freq.keys(), wordle_freq.values(), color="skyblue")
    axs[0, 0].set_title("Wordle Letter Frequency")
    axs[0, 0].set_xlabel("Letters")
    axs[0, 0].set_ylabel("Frequency")

    # Plot 2: Wordle Positional Frequency Heatmap
    heatmap_data_wordle = pd.DataFrame(wordle_positional_freq).fillna(0)
    sns.heatmap(heatmap_data_wordle, cmap="Blues", annot=True, fmt=".0f", ax=axs[0, 1])
    axs[0, 1].set_title("Wordle Positional Frequency")

    # Plot 3: English Letter Frequency
    axs[1, 0].bar(english_freq.keys(), english_freq.values(), color="lightgreen")
    axs[1, 0].set_title("English Five-Letter Words Letter Frequency")
    axs[1, 0].set_xlabel("Letters")
    axs[1, 0].set_ylabel("Frequency")

    # Plot 4: English Positional Frequency Heatmap
    heatmap_data_english = pd.DataFrame(english_positional_freq).fillna(0)
    sns.heatmap(
        heatmap_data_english, cmap="Greens", annot=True, fmt=".0f", ax=axs[1, 1]
    )
    axs[1, 1].set_title("English Positional Frequency")

    # Adjust layout
    plt.tight_layout()
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

        # Step 5: Visualize Frequencies
        print("Visualizing data...")
        visualize_data(
            wordle_freq, wordle_positional_freq, english_freq, english_positional_freq
        )

        print("Process completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


# Entry Point
if __name__ == "__main__":
    main()
