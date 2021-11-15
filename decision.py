import math
from entropy_helper import calculate_information_gain_for_categorical_feature, \
    calculate_information_gain_for_numerical_feature
import operator
from important_classes import Record, NodeState


def decision_tree_train(data, feature_names, feature_types, entropy_cutoff=0, max_depth=-1):
    stack = []
    target_label_count = {}
    for record in data:
        if record.output not in target_label_count:
            target_label_count[record.output] = 1.0
        else:
            target_label_count[record.output] += 1.0
    total_entropy = 0.0
    n = len(data)
    for label, label_count in target_label_count.items():
        total_entropy += (-(label_count / n) * math.log2(label_count / n))

    root = NodeState(data, feature_names, target_label_count, total_entropy, 1)
    stack.append(root)
    while len(stack) != 0:
        node = stack.pop()
        if max_depth != -1 and node.current_depth > max_depth:
            break
        if node.current_entropy > entropy_cutoff:  # generate children only if necessary entropy is not met
            best_information_gain = 0.0
            best_feature = None
            best_splitting_value = None
            best_entropy_feature_label = None
            for feature in node.feature_names:
                current_information_gain = 0.0
                splitting_value = None
                if feature_types[feature] == 'C':
                    current_information_gain, entropy_feature_label, splitting_value = calculate_information_gain_for_categorical_feature(
                        node, feature)
                else:
                    #  min_split_ratio = 20  # 1:20 ratio of smallest branch to initial total data size
                    current_information_gain, entropy_feature_label, splitting_value = calculate_information_gain_for_numerical_feature(
                        node, feature)
                if current_information_gain > best_information_gain:
                    best_information_gain = current_information_gain
                    best_feature = feature
                    best_entropy_feature_label = entropy_feature_label
                    best_splitting_value = splitting_value  # None if best feature is categorical
            if best_feature is not None:
                stack.extend(list(node.generate_children(best_feature, best_splitting_value,
                                                         best_entropy_feature_label).values()))  # putting children of current node in stack
    return root


def decision_tree_predict(model, record):
    node = model
    while len(node.children) != 0:
        if node.splitting_feature_type == 'C':
            prop = record.features[node.splitting_feature]
        elif record.features[node.splitting_feature] > node.splitting_value:
            prop = 'greater than'
        else:
            prop = 'lesser than'
        node = node.children[prop]
    majority_label = max(node.target_label_count.items(), key=operator.itemgetter(1))[0]
    return majority_label
