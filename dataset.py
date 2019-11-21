import numpy as np
import jieba
from collections import Counter
import os


class DataSet(object):
    def __init__(self, data_dir, stop_words_file, label_file, vocab_dir=None, vocab_list=None, remove_vocab=True, split='	', is_trainng=True):
        self.data_dir = data_dir
        self.stop_words_file = stop_words_file
        self.remove_vocab = remove_vocab
        self.label_file = label_file
        self.split = split
        self.idx_to_textlabel = dict()
        self.data_set = []
        self.vocab_dir = vocab_dir
        if is_trainng:
            self.vocab_list = []
        else:
            assert vocab_list is not None, "missing vocab_list"
            self.vocab_list = vocab_list
        self.is_training = is_trainng

    def get_start(self):
        #  获取标签集合
        with open(self.label_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            self.idx_to_textlabel[idx] = line.strip()
        with open(self.stop_words_file, encoding="utf-8") as sf:
            lines = sf.readlines()
        # 获取停用词
        stop_words = set([line.strip() for line in lines])
        stop_words.add(' ')
        stop_words.add('\n')
        stop_words.add('')
        examples = []
        # 处理数据
        with open(self.data_dir, encoding="utf-8") as f1:
            for record in f1:
                text, label = record.strip().split(self.split)
                text = text.strip()
                label = int(label)
                words = list(jieba.cut(text))
                words = [word for word in words if word not in stop_words]
                examples.append({"text": words, "label": label})   # 单词，标签
        if self.is_training:
            self.writeVocab(examples)

        self.make_dataset(examples)


    def make_dataset(self, examples):
        for example in examples:
            words

    def writeVocab(self, examples):
        words = Counter()  # 定义一个计数器。键是词，值是这个词的频次
        for e in examples:
            words.update(e["text"])
        vocab_file = os.path.join(self.vocab_dir, "vocab.txt")
        if self.remove_vocab and os.path.isfile(vocab_file):
            os.remove(vocab_file)
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in words.most_common():
                f.write(word[0]+'\n')
                self.vocab_list.append(word)

    def get_vocab_list(self):
        return self.vocab_list


