import glob
import json
import jieba
import random


def read_all_data(title_name):
    for fname in glob.iglob(
        '/home/hanjing/workspace/ein-doc-reader/more_data_dir/*.json',
        recursive=False
    ):
        prod_data = json.load(open(fname))
        extractions = prod_data[0]['terms']
        all_extract_items = []
        for item in extractions:
            for extract_item in item['units']:
                all_extract_items.append(extract_item)
        for item in all_extract_items:
            if not isinstance(item, dict):
                continue
            if not item['title'] == title_name:
                continue
            if 'para_content' not in item:
                continue
            extract_content = item['content']
            raw_content = item['para_content']
            yield extract_content, raw_content


def tag_data(title_name):
    tag_results = []  # [sequences, tags]
    all_cut_words = []
    for extract_content, raw_content in read_all_data(title_name):
        start_pos = raw_content.find(extract_content)
        tags = []
        cut_words = []
        if start_pos < 0:
            print('Warning extract content: %s not find in raw_content: %s'
                  % (extract_content, raw_content))
            continue
        cut_raw_content = list(jieba.tokenize(raw_content))
        end_pos = start_pos + len(extract_content)
        # for idx in range(len(raw_content)):
        for cut_item in cut_raw_content:
            word, word_start, word_end = cut_item
            if word_end - 1 < start_pos or word_start > end_pos - 1:
                tags.append('O-' + title_name)
            elif word_start <= start_pos and word_end > start_pos:
                tags.append('B-' + title_name)
            elif word_end >= end_pos and word_start <= end_pos - 1:
                tags.append('E-' + title_name)
            else:
                tags.append('M-' + title_name)
            cut_words.append(word)

        assert(len(tags) == len(cut_words))

        if cut_words in all_cut_words:
            continue

        tag_results.append((cut_words, tags))
        all_cut_words.append(cut_words)

    return tag_results


def write_to_file(output_file, data):
    with open(output_file, 'w') as output_fn:
        for sequences, tags in data:
            for sequence, tag in zip(sequences, tags):
                if sequence == '\n':
                    sequence = '\\n'
                output_fn.write('%s\t%s\n' % (sequence, tag))
            # 训练数据分割符
            output_fn.write('########\n')


def main():
    title_names = ['合同构成', '投保范围', '责任免除情形一', '犹豫期后解除合同', '什么是犹豫期',
                   '因重大过失未履行如实告知义务', '什么是宽限期', '保险事故发生后未及时通知的处理',
                   '委托他人办理申领保险金需提供的材料', '保险金作为被保险人遗产时需提供的材料']
    train_data = []
    test_data = []
    for title in title_names:
        data = tag_data(title)
        train_data.extend(data[:int(len(data) * 0.8)])
        test_data.extend(data[int(len(data) * 0.8):])
    random.shuffle(train_data)
    random.shuffle(test_data)
    write_to_file('train_data.txt', train_data)
    write_to_file('test_data.txt', test_data)


if __name__ == '__main__':
    main()
