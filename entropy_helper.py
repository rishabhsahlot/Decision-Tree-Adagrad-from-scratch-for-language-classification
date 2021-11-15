import math


def calculate_information_gain_for_categorical_feature(node, feature):
    feature_label_target_label_count = {}
    feature_label_count = {}
    for record in node.data:
        if record.features[feature] not in feature_label_count:
            feature_label_count[record.features[feature]] = 0  # Count of each label in the feature under consideration
            feature_label_target_label_count[record.features[
                feature]] = {}  # For each label of target feature count of each label in the feature under consideration
        feature_label_count[record.features[feature]] += 1.0
        if record.output not in feature_label_target_label_count[record.features[feature]]:
            feature_label_target_label_count[record.features[feature]][record.output] = 1.0
        else:
            feature_label_target_label_count[record.features[feature]][record.output] += 1.0
    information_gain = node.current_entropy
    entropy_feature_label = {}
    for feature_label in feature_label_target_label_count:
        entropy_feature_label[feature_label] = 0.0
        for target_label in feature_label_target_label_count[feature_label]:
            probability = feature_label_target_label_count[feature_label][target_label] / feature_label_count[
                feature_label]
            entropy_feature_label[feature_label] -= probability * math.log2(probability)
        probability = feature_label_count[feature_label] / len(node.data)
        information_gain -= probability * entropy_feature_label[feature_label]
    return information_gain, entropy_feature_label, None


def calculate_information_gain_for_numerical_feature(node, feature):  # , min_split_ratio):
    node.data.sort(key=lambda r: r.features[feature])
    #  min_no_samples = (len(node.data))//min_split_ratio
    split_index = 1
    best_information_gain = -1
    best_splitting_index = -1
    best_splitting_value = 0
    entropy_dict = {'lesser than': 0, 'greater than': 0}
    current_target_label_count = {}
    for target_label in node.target_label_count:
        current_target_label_count[target_label] = 0
    current_target_label_count[node.data[0].output] = 1
    while split_index < len(node.data)-1:
        #current_target_label_count[node.data[split_index].output] += 1
        while split_index < len(node.data)-1 and node.data[split_index-1].features[feature] == node.data[split_index].features[feature]:
            current_target_label_count[node.data[split_index].output] += 1
            split_index += 1
        if split_index < len(node.data)-1:
            entropy_lesser_than = 0.0
            entropy_greater_than = 0.0
            for target_label, count in current_target_label_count.items():
                if count != 0:
                    probability = count / (split_index)
                    entropy_lesser_than -= probability * math.log2(probability)
                if (node.target_label_count[target_label] - count) != (len(node.data) - split_index) and node.target_label_count[target_label] != count:
                    probability = (node.target_label_count[target_label] - count) / (len(node.data) - split_index)
                    entropy_greater_than -= probability * math.log2(probability)
            probability = (split_index) / len(node.data)
            #  vi = -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)
            information_gain = (node.current_entropy - (
                    probability * entropy_lesser_than + (1 - probability) * entropy_greater_than))  # / vi
            if best_information_gain < information_gain:
                best_information_gain = information_gain
                best_splitting_value = (node.data[split_index-1].features[feature] + node.data[split_index].features[
                    feature]) / 2
                best_splitting_index = split_index
                entropy_dict = {"lesser than": entropy_lesser_than, "greater than": entropy_greater_than}
            current_target_label_count[node.data[split_index].output] += 1
            split_index += 1
    # now we calculate final info-gain, for the value
    # prob = best_splitting_index / len(node.data)
    # best_information_gain = node.current_entropy - prob * entropy_dict["lesser than"] - (1 - prob) * entropy_dict[
    #     "greater than"]

    return best_information_gain, entropy_dict, best_splitting_value
