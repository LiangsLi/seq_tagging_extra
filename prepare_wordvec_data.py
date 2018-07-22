import glob
import json
import jieba
import pickle


def read_all_data():
    raw_text = []
    for fname in glob.iglob(
        '/Users/han/workspace/work/neural-bot/neural_bot/sequence_tagging/insurance_data/*.json',
        recursive=False
    ):
        prod_data = json.load(open(fname))
        extractions = prod_data[0]['raw_content']
        raw_text.append(extractions)
    return raw_text


def cut_data(all_raw_text):
    all_text = []
    for text in all_raw_text:
        all_text.append(list(jieba.cut(text)))
    return all_text


def main():
    raw_text = read_all_data()
    cut_text = cut_data(raw_text)
    with open('cut_text.pickle', 'wb') as pickle_file:
        print(cut_text)
        pickle.dump(cut_text, pickle_file)


if __name__ == '__main__':
    main()
