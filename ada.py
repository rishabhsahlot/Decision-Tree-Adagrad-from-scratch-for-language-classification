import math
import random
import bisect
from decision import decision_tree_train, decision_tree_predict
import operator


def adaboost_train(data, feature_names, feature_type, no_of_stumps):
    stumps = []
    for __ in range(no_of_stumps):
        current_stump = decision_tree_train(data, feature_names, feature_type, 0, 1)
        if current_stump.splitting_feature is None:
            return stumps
        error_count = 0
        child_max_target_label = {}
        for child_feature_label, child in current_stump.children.items():
            max_label = list(child.target_label_count.keys())[0]
            for label, count in child.target_label_count.items():
                if count > child.target_label_count[max_label]:
                    max_label = label
            child_max_target_label[child_feature_label] = max_label
            error_count += (len(child.data)-child.target_label_count[max_label])
        error_rate = error_count/len(data)
        epsilon = 1 / len(data)
        alpha = 0.5 * math.log((1-error_rate+epsilon)/(error_rate+epsilon), 10)  # for smoothening
        current_stump.significance = alpha
        stumps.append(current_stump)
        original_weight = 1/len(data)
        correct_weight_multiplier = math.e**(-alpha)
        wrong_weight_multiplier = math.e ** alpha
        new_weights = []
        for record in data:
            if current_stump.splitting_feature_type == 'C':
                prop = record.features[current_stump.splitting_feature]
            elif record.features[current_stump.splitting_feature] > current_stump.splitting_value:
                prop = 'greater than'
            else:
                prop = 'lesser than'
            if record.output != child_max_target_label[prop]:  # wrongly predicted
                new_weights.append(original_weight*wrong_weight_multiplier)
            else:  # correctly predicted
                new_weights.append(original_weight*correct_weight_multiplier)
        total_weight = sum(new_weights)
        new_weights = list(map(lambda w: w / total_weight, new_weights))  # normalize
        for i in range(1,len(new_weights)):
            new_weights[i]+=new_weights[i-1]
        #  new_weights = [new_weights[i-1]+new_weights[i] for i in range(1, len(new_weights))]  # cumulative        new_weights[-1] = 1  # round last value to 1
        new_weights[-1]=1
        new_data = []
        for __ in range(len(data)):
            prob = random.random()  # generate random probability value
            index = bisect.bisect_left(new_weights, prob)  # binary search on cumulative probabilities
            new_data.append(data[index])  # append appropriate value to new data
        data = new_data
    return stumps


def adaboost_predict(model, record):
    y = 0
    for stump in model:
        majority_label = decision_tree_predict(stump, record)
        if majority_label == 'en':
            y += stump.significance
        else:
            y -= stump.significance
    if y > 0:
        return 'en'
    else:
        return 'nl'
