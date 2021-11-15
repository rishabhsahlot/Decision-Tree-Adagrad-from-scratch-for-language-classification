import sys
import math
import pickle
from decision import Record, decision_tree_train, decision_tree_predict
from ada import adaboost_train, adaboost_predict
from features import ascii_checker, longest_word_length, c_percentage, p_percentage, longest_vowel_streak, \
    longest_consonant_streak, stop_words_checker, average_word_length, average_letter_distribution


def generate_record(words, dutch_stop_words, eng_stop_words, output=None):
    feature_val = {}
    feature_val['ascii checker'] = ascii_checker(words)
    feature_val['longest word length'] = longest_word_length(words)
    feature_val['c percentage'] = c_percentage(words)
    feature_val['p percentage'] = p_percentage(words)
    feature_val['longest vowel streak'] = longest_vowel_streak(words)
    feature_val['longest consonant streak'] = longest_consonant_streak(words)
    feature_val['has english stop words'] = stop_words_checker(words, eng_stop_words)
    feature_val['has dutch stop words'] = stop_words_checker(words, dutch_stop_words)
    feature_val['average word length'] = average_word_length(words)
    feature_val['average letter'] = average_letter_distribution(words)
    return Record(feature_val, output)


def fill_table(path, dutch_path, eng_path):
    feature_type = {'ascii checker': 'C', 'longest word length': 'N',
                    'c percentage': 'N', 'p percentage': 'N',
                    'longest vowel streak': 'N', 'longest consonant streak': 'N',
                    'has english stop words': 'C',
                    'has dutch stop words': 'C',
                    'average word length': 'N',
                    'average letter': 'N'}  # to be initialized here
    data = []
    fp_dutch_stop_words = open(dutch_path, 'r', encoding='utf-8')
    fp_eng_stop_words = open(eng_path, 'r', encoding='utf-8')
    dutch_stop_words = set([word[:-1] for word in fp_dutch_stop_words.readlines()])
    eng_stop_words = set([word[:-1] for word in fp_eng_stop_words.readlines()])
    fp_eng_stop_words.close()
    fp_dutch_stop_words.close()
    union = dutch_stop_words.union(eng_stop_words)
    dutch_stop_words = union.difference(eng_stop_words)
    eng_stop_words = union.difference(dutch_stop_words)
    fp = open(path, 'r', encoding='utf-8')
    line = fp.readline()[:-1].split('|')
    while len(line[0]) != 0:
        if len(line) != 1:
            record = generate_record(line[1].split(' '), dutch_stop_words, eng_stop_words, line[0])
        else:
            record = generate_record(line[0].split(" "), dutch_stop_words, eng_stop_words)
        data.append(record)
        line = fp.readline()[:-1].split('|')
    feature_names = list(data[0].features.keys())
    return data, feature_names, feature_type


def obj_to_dict(obj):
    return obj.__dict__


# D:\\Spring2020\\FoundationsofAI\\Dataset\\train.txt
def train(examples, hypothesisOut, learning_type, dutch_path='dutch_stopwords.txt', eng_path='english_stopwords.txt'):
    data, feature_names, feature_type = fill_table(examples, dutch_path, eng_path)
    root = None
    if learning_type == 'ada':
        no_of_estimators = 30
        model = adaboost_train(data, feature_names, feature_type, no_of_estimators)
    else:
        accuracy_in_training_data_set = 0.99
        entropy_cutoff = -accuracy_in_training_data_set * math.log2(accuracy_in_training_data_set)
        model = decision_tree_train(data, feature_names, feature_type, entropy_cutoff)
    with open(hypothesisOut, "wb") as hypothesis:
        pickle.dump(model, hypothesis)


def predict(hypothesis, test_file, predictions='prediction.txt', dutch_path='dutch_stopwords.txt',
            eng_path='english_stopwords,txt'):
    data, feature_names, feature_type = fill_table(test_file, dutch_path, eng_path)
    with open(hypothesis, 'rb') as json_file:
        m = pickle.load(json_file)
    if type(m) is list:  # model is for adaboost
        predicted_labels = []
        for record in data:
            predicted_label = adaboost_predict(m, record)
            predicted_labels.append(predicted_label + "\n")
    else:  # model is for decision trees
        predicted_labels = []
        for record in data:
            predicted_label = decision_tree_predict(m, record)
            predicted_labels.append(predicted_label + "\n")
    print(predicted_labels)
    with open(predictions, 'w') as f:
        f.writelines(predicted_labels)


def check_accuracy(actual_label_path, predicted_label_path):
    with open(actual_label_path, 'rb') as fp:
        actual_labels = [label[:-1] for label in fp.readlines()]
    with open(predicted_label_path, 'rb') as fp:
        predicted_labels = [label[:-1] for label in fp.readlines()]
    print(len(actual_labels))
    print(len(predicted_labels))
    wrong_predictions = {}
    for i in range(len(actual_labels)):
        if actual_labels[i] != predicted_labels[i]:
            if predicted_labels[i] in wrong_predictions.keys():
                wrong_predictions[predicted_labels[i]] += 1.0
            else:
                wrong_predictions[predicted_labels[i]] = 1.0
    print("Accuracy = %f\n" % (100.0 * (len(actual_labels) - sum(wrong_predictions.values())) / len(actual_labels)))
    for label, value in wrong_predictions.items():
        print("False %s = %d" % (label, int(value)))


def main():
    work_to_be_done = sys.argv[1]
    if work_to_be_done == 'training':
        path = sys.argv[2]
        hypothesisOut = sys.argv[3]
        algorithm = sys.argv[4]
        if len(sys.argv) == 7:
            dutch_path = sys.argv[5]
            eng_path = sys.argv[6]
        else:
            dutch_path = 'dutch_stopwords.txt'
            eng_path = 'english_stopwords.txt'
        train(path, hypothesisOut, algorithm, dutch_path, eng_path)
    elif work_to_be_done == 'testing':
        modelpath = sys.argv[2]
        testpath = sys.argv[3]
        if len(sys.argv) > 4:
            predictions = sys.argv[4]
        else:
            predictions = 'prediction.txt'
        if len(sys.argv) == 7:
            dutch_path = sys.argv[5]
            eng_path = sys.argv[6]
        else:
            dutch_path = 'dutch_stopwords.txt'
            eng_path = 'english_stopwords.txt'
        predict(modelpath, testpath, predictions, dutch_path, eng_path)
    else:
        actual_label_path = sys.argv[2]
        predicted_label_path = sys.argv[3]
        check_accuracy(actual_label_path, predicted_label_path)


if __name__ == '__main__':
    main()

#  testing dtmodel2.txt D:\\Spring2020\\FoundationsofAI\\Dataset\\test.txt predictdtmodel2.txt D:\\Spring2020\\FoundationsofAI\\Dataset\\dutch_stopwords.txt D:\\Spring2020\\FoundationsofAI\\Dataset\\english_stopwords.txt
