import math
#from scipy.stats import chisquare


def ascii_checker(words):
    weird_ascii_letter_count = 0
    for word in words:
        for letter in word:
            if ord(letter) > ord('z'):
                weird_ascii_letter_count += 1
    if weird_ascii_letter_count > 0:
        return "True"
    else:
        return "False"


def longest_word_length(words):
    return max(map(lambda x: len(x), words))


def c_percentage(words):
    ct = 0
    ct_total = 0
    for word in words:
        for letter in word:
            if letter == 'c':
                ct += 1
        ct_total += len(word)
    return ct / ct_total


def p_percentage(words):
    ct = 0
    ct_total = 0
    for word in words:
        for letter in word:
            if letter == 'p':
                ct += 1
        ct_total += len(word)
    return ct / ct_total


def longest_vowel_streak(words):
    ans = 0
    for word in words:
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[j] in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                j += 1
            ans = max(ans, j - i)
            i = j + 1
    return ans


def longest_consonant_streak(words):
    ans = 0
    for word in words:
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[j] not in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                j += 1
            ans = max(ans, j - i)
            i = j + 1
    return ans


def stop_words_checker(words, stop_words):
    for word in words:
        if word.lower() in stop_words:
            return "True"
    return "False"


def average_word_length(words):
    return sum(map(len, words)) / len(words)


def average_letter_distribution(words):  # averages the ordinal values of letters A-Z
    ct = 0
    tot = 0
    for word in words:
        for letter in word:
            if letter.isalpha():
                tot += (ord(letter.lower())-ord('a'))
                ct += 1
    return tot // ct


# def one_gram_chi_test(words,expected_letter_freq_dutch, expected_letter_freq_eng):
#     observed_letter_freq = [0] * len(expected_letter_freq_eng)
#     for word in words:
#         for letter in words:
#             if letter.isalpha():
#                 observed_letter_freq[ord(letter.lower()) - ord('a')] += 1
#     min_val_eng , min_val_dutch = min(expected_letter_freq_eng.values()), min(expected_letter_freq_eng.values())
#     mult_eng, mult_dutch = 5 / min_val_eng, 5 / min_val_dutch
#     for key in expected_letter_freq_eng.keys():
#         expected_letter_freq_eng[key] = math.ceil(expected_letter_freq_eng[key]*mult_eng)
#     for key in expected_letter_freq_eng.keys():
#         expected_letter_freq_dutch[key] = math.ceil(expected_letter_freq_dutch[key] * mult_dutch)
#     chi_eng, p_eng = chisquare(observed_letter_freq,f_exp=expected_letter_freq_eng)
#     chi_dutch, p_dutch = chisquare(observed_letter_freq,f_exp=expected_letter_freq_dutch)
#     return p_eng-p_dutch
#
#
# def two_gram_chi_test(words, expected_2gram_freq_dutch, expected_2gram_freq_eng):
#     observed_2gram_freq = [0] * len(expected_2gram_freq_eng)
#     for word in words:
#         for i in range(len(word)-1):
#             if word[i:i+2].isalpha():
#                 observed_2gram_freq[word[i:i+2]] += 1
#     min_val_eng, min_val_dutch = min(expected_letter_freq_eng.values()), min(expected_letter_freq_eng.values())
#     mult_eng, mult_dutch = 5 / min_val_eng, 5 / min_val_dutch
#     for key in expected_letter_freq_eng.keys():
#         expected_letter_freq_eng[key] = math.ceil(expected_letter_freq_eng[key] * mult_eng)
#     for key in expected_letter_freq_eng.keys():
#         expected_letter_freq_dutch[key] = math.ceil(expected_letter_freq_dutch[key] * mult_dutch)
#     chi_eng, p_eng = chisquare(observed_letter_freq, f_exp=expected_letter_freq_eng)
#     chi_dutch, p_dutch = chisquare(observed_letter_freq, f_exp=expected_letter_freq_dutch)
#     return p_eng - p_dutch