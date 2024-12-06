from wordfreq import word_frequency
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from main import read_from_file


def get_word_frequencies(word_list, lang="en"):
    """
    Calculate word frequencies for a given word list using the wordfreq library.

    Args:
        word_list (list): List of words to calculate frequencies for.
        lang (str): Language for word frequency calculation (default is 'en').

    Returns:
        pd.DataFrame: A dataframe with words and their frequencies.
    """
    frequencies = {word: word_frequency(word, lang) for word in word_list}
    print(pd.DataFrame(frequencies.items(), columns=["Word", "Frequency"]))
    return pd.DataFrame(frequencies.items(), columns=["Word", "Frequency"])


wordle_list = read_from_file("wordle.txt")
five_letter_words = read_from_file("five_letter_words.txt")

wordle_frequencies_df = get_word_frequencies(wordle_list)
nltk_frequencies_df = get_word_frequencies(five_letter_words)
