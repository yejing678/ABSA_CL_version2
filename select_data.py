import os
from transformers import pipeline


def sentiment_analyzer(data):
    analyzer = pipeline("sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentence_list = []
    aspect_list = []
    aspect_label = []
    sentence_label_list = []
    sentence_labels = []
    for i in range(0, len(data), 3):
        sentence = data[i]
        aspect = data[i + 1]
        aspect_polarity = data[i + 2]
        sentence_list.append(sentence)
        aspect_list.append(aspect)
        aspect_label.append(aspect_polarity)
    results = analyzer(sentence_list)
    d = {
        '1 star': '-1',
        '2 stars': '-1',
        '3 stars': '0',
        '4 stars': '1',
        '5 stars': '1',
    }
    for result in results:
        sentence_label = result['label']
        sentence_label_list.append(d[sentence_label])
    for i in range(len(sentence_label_list)):
        sentence_labels.append(sentence_list[i])
        sentence_labels.append(aspect_list[i])
        sentence_labels.append(aspect_label[i])
        sentence_labels.append(sentence_label_list[i])
    return aspect_label, sentence_label_list, sentence_labels


def select_neu(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    data = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        if polarity == '0':
            data.append(text)
            data.append(aspect)
            data.append(polarity)
    return data


def select_neu_label(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    data = []
    for i in range(0, len(lines), 3):
        text= lines[i].strip()
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        if polarity == '0':
            sentiment_label = '1'
        else:
            sentiment_label = '0'
        data.append(text)
        data.append(aspect)
        data.append(polarity)
        data.append(sentiment_label)
    return data


def read_data(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    data = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        data.append(text)
        data.append(aspect)
        data.append(polarity)
    return data


def analysis_consistency(aspect_label_list, sentence_label_list):
    count = 0
    total_num = len(sentence_label_list)
    for i in range(len(sentence_label_list)):
        if sentence_label_list[i] == aspect_label_list[i]:
            count += 1
    consistency_rate = count / total_num
    return count, consistency_rate


if __name__ == '__main__':
    fname = "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/train_total_text.txt"
    base_path = "/home/disk2/jye/ABSA/datasets/implicit/laptop/"
    # print('read file...')
    # data = read_data(fname)
    print('select data...')
    data = select_neu_label(fname)
    print('write selected_neu_txt...')
    save_selected_data_path = os.path.join(base_path, 'txt_selected_neu')
    if not os.path.exists(save_selected_data_path):
        os.makedirs(save_selected_data_path)
    with open(os.path.join(save_selected_data_path, 'train.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            f.write(data[i] + '\n')
    f.close()

    # print('analyzing data...')
    # aspect_label_list, sentence_label_list, sentence_labels = sentiment_analyzer(data)
    # print('writing  sentiment_labeled_txt...')
    # save_sentence_labeled_path = os.path.join(base_path, 'sentiment_labeled')
    # if not os.path.exists(save_sentence_labeled_path):
    #     os.makedirs(save_sentence_labeled_path)
    # with open(os.path.join(save_sentence_labeled_path, 'train_implicit_text.txt'), 'w', encoding='utf-8') as f_1:
    #     for i in range(len(data)):
    #         f_1.write(sentence_labels[i] + '\n')
    # f_1.close()
    #
    # print('analysis consistency...')
    # consistency_num, consistency_rate = analysis_consistency(aspect_label_list, sentence_label_list)
    # print('consistency_num:', consistency_num)
    # print('consistency_rate:', consistency_rate)
    print('completed')
