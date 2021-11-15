
class Record:
    def __init__(self, features, output_value=None):
        self.features = features
        self.output = output_value


class NodeState:
    def __init__(self, data, feature_names, target_label_count, entropy, cur_depth):
        self.data = data
        self.feature_names = feature_names
        self.target_label_count = target_label_count
        self.splitting_feature = None
        self.splitting_feature_type = 'C'
        self.splitting_value = None
        self.current_entropy = entropy
        self.current_depth = cur_depth
        self.significance = None  # set only in ada boost
        self.children = {}  # if numerical contains 'less_than' & 'greater_than' value, otherwise categorical label values

    def generate_children(self, feature, splitting_value, entropy_label):
        self.splitting_feature = feature

        if splitting_value is None:  # Categorical feature
            self.splitting_feature_type = 'C'
            new_feature_names = list(set(self.feature_names) - {feature})
            split_data = {}
            new_target_label_count = {}
            for record in self.data:
                if record.features[feature] not in split_data:
                    split_data[record.features[feature]] = []
                    new_target_label_count[record.features[feature]] = {}
                if record.output not in new_target_label_count[record.features[feature]]:
                    new_target_label_count[record.features[feature]][record.output] = 1
                else:
                    new_target_label_count[record.features[feature]][record.output] += 1
                split_data[record.features[feature]].append(record)
            for label in split_data:
                self.children[label] = NodeState(split_data[label], new_feature_names, new_target_label_count[label],
                                                 entropy_label[label], self.current_depth + 1)
        else:  # Numerical feature
            self.splitting_feature_type = 'N'
            self.splitting_value = splitting_value
            split_data = {"lesser than": [], "greater than": []}
            new_target_label_count = {"lesser than": {}, "greater than": {}}
            for record in self.data:
                if record.features[feature] < splitting_value:
                    prop = "lesser than"
                else:
                    prop = "greater than"
                if record.output not in new_target_label_count[prop]:
                    new_target_label_count[prop][record.output] = 1
                else:
                    new_target_label_count[prop][record.output] += 1
                split_data[prop].append(record)
            for label in ["lesser than", "greater than"]:
                self.children[label] = NodeState(split_data[label], self.feature_names, new_target_label_count[label],
                                                 entropy_label[label], self.current_depth + 1)
        return self.children

    # def __str__(self):
    #     return self.splitting_feature + ":" + self.splitting_value + "\nEntropy:" + str(
    #         self.current_entropy) + "\ndepth:" + str(self.current_depth)
