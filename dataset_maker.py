import sys


def testing_maker(dataset_input, dataset_language, dataset_path, dataset_output_path):
    with open(dataset_input, 'r',encoding='utf-8') as input_file:
        lines = input_file.readlines()
        with open(dataset_path, 'a',encoding='utf-8') as output:
            with open(dataset_output_path, 'a',encoding='utf-8') as output_path:
                lines= [line.split() for line in lines]
                fifteen_words = []
                for i in range(len(lines)):
                    for j in range(len(lines[i])):
                        fifteen_words.append(lines[i][j])
                        if len(fifteen_words) == 15:
                            output.write(" ".join(fifteen_words)+"\n")
                            output_path.write(dataset_language+"\n")
                            fifteen_words = []


def training_maker(dataset_input, dataset_language, dataset_path):
    with open(dataset_input, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        with open(dataset_path, 'a', encoding='utf-8') as output:
            lines= [line.split() for line in lines]
            fifteen_words = []
            for i in range(len(lines)):
                for j in range(len(lines[i])):
                    fifteen_words.append(lines[i][j])
                    if len(fifteen_words) == 15:
                        output.write(dataset_language+"|"+" ".join(fifteen_words)+"\n")
                        fifteen_words = []


def main():
    dataset_type = sys.argv[1]
    dataset_input = sys.argv[2]
    dataset_language = sys.argv[3]
    dataset_path = sys.argv[4]
    if dataset_type == 'training':
        training_maker(dataset_input, dataset_language, dataset_path)
    else:
        dataset_output_path = sys.argv[5]
        testing_maker(dataset_input, dataset_language, dataset_path, dataset_output_path)

if __name__ == '__main__':
    main()