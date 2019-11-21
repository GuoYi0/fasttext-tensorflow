import numpy as np
import jieba
from collections import Counter
import os
from random import shuffle
from tqdm import tqdm
from multiprocessing import Queue, Process


class DataSet(object):
    def __init__(self, data_dir, stop_words_file, label_file, vocab_dir=None, vocab_list=None,
                 remove_vocab=True, split='	', is_trainng=True, batch_size=1, *, max_words_length):
        """
        :param data_dir:  数据集路径， 一个txt文档
        :param stop_words_file:  停用词路径，一个txt文档
        :param label_file:  标签文件，一个txt文档
        :param vocab_dir:  词汇表路径，一个文件夹地址
        :param vocab_list:  词汇表。训练时候可以为None，由训练集来生成。验证和测试的时候不能为None
        :param remove_vocab:  是否移除词汇表，从而进行更新
        :param split:  # 数据集中文本和标签之间的分隔符
        :param is_trainng:  # 是否在训练
        :param batch_size:  # 每次迭代返回的batch_size
        :param max_words_length:  # 每个文本最大的词汇个数
        """
        self.data_dir = data_dir
        self.remove_vocab = remove_vocab
        self.split = split
        self.idx_to_textlabel = {}  # 索引到标签的映射
        self.vocab_dir = vocab_dir
        if is_trainng and vocab_list is None:
            self.vocab_list = []
        else:
            assert vocab_list is not None, "missing vocab_list"
            self.vocab_list = vocab_list
        self.is_training = is_trainng
        self.max_words_length = max_words_length
        self._cur = 0
        self.batch_size = batch_size
        self.data_set = None
        self.data_idx = None
        self.num_classes = None
        self.vocab_size = 0
        self.get_start(label_file, stop_words_file)  # 准备数据集

    def get_start(self, label_file, stop_words_file):
        #  获取标签集合
        with open(label_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            self.idx_to_textlabel[idx] = line.strip()
        self.num_classes = len(self.idx_to_textlabel)
        with open(stop_words_file, encoding="utf-8") as sf:
            lines = sf.readlines()
        # 获取停用词
        stop_words = set([line.strip() for line in lines])
        stop_words.add(' ')
        stop_words.add('\n')
        stop_words.add('')
        examples = []
        # 处理数据， 得到单词列表和标签
        with open(self.data_dir, encoding="utf-8") as f1:
            for record in f1:
                text, label = record.strip().split(self.split)
                text = text.strip()
                label = int(label)
                words = list(jieba.cut(text))
                words = [word for word in words if word not in stop_words]
                examples.append({"words": words, "label": label})  # 单词，标签
        # 训练时候才需要填充词汇表
        if self.is_training:
            self.writeVocab(examples)
        self.make_dataset(examples)  # 制作训练集

    def _worker(self, s, q):
        for e in s:
            word_ids = []
            for word in e["words"]:
                # 若 word 在self.vocab_list之中，就返回其索引位置加1，否则返回0
                try:
                    word_ids.append(self.vocab_list.index(word) + 1)
                except ValueError:
                    word_ids.append(0)
            # 如果词汇个数太多，则优先删除0，删除0以后依然太多；则随机选择self.max_words_length个元素
            if len(word_ids) > self.max_words_length:
                word_ids = [w for w in word_ids if w != 0]
            if len(word_ids) > self.max_words_length:
                shuffle(word_ids)
                word_ids = word_ids[:self.max_words_length]
            q.put_nowait({"word_ids": word_ids, "label": e["label"]})

    def make_dataset(self, examples):
        """
        制作训练集，得到 单词ids和对应的文本label
        :param examples:
        :return:
        """
        data_set = []
        num_example = len(examples)
        num_threads = 12
        num_example_pre_thread = num_example / num_threads
        result_queue = Queue(100000)
        procs = []
        pbar = tqdm(total=num_example)
        for i in range(num_threads):
            start = int(i * num_example_pre_thread)
            end = min(int((i + 1) * num_example_pre_thread), num_example)
            split = examples[start:end]
            proc = Process(target=self._worker, args=(split, result_queue))
            proc.start()
            procs.append(proc)
        for i in range(num_example):
            t = result_queue.get()
            data_set.append(t)
            pbar.update()
        for p in procs:
            p.join()

        self.data_set = tuple(data_set)  # 考虑性能需求，把数据集用tuple保存下来
        self.data_idx = list(range(len(self.data_set)))
        del data_set
        if self.is_training:
            shuffle(self.data_idx)

    def make_dataset2(self, examples):
        """
        制作训练集，得到 单词ids和对应的文本label
        :param examples:
        :return:
        """
        data_set = []
        print("making dataset ...")
        for example in tqdm(examples):
            word_ids = []
            for word in example["words"]:
                # 若 word 在self.vocab_list之中，就返回其索引位置加1，否则返回0
                try:
                    word_ids.append(self.vocab_list.index(word) + 1)
                except ValueError:
                    word_ids.append(0)
            # 如果词汇个数太多，则优先删除0，删除0以后依然太多；则随机选择self.max_words_length个元素
            if len(word_ids) > self.max_words_length:
                word_ids = [w for w in word_ids if w != 0]
            if len(word_ids) > self.max_words_length:
                shuffle(word_ids)
                word_ids = word_ids[:self.max_words_length]
            data_set.append({"word_ids": word_ids, "label": example["label"]})
            # from IPython import embed; embed()
        self.data_set = tuple(data_set)  # 考虑性能需求，把数据集用tuple保存下来
        self.data_idx = list(range(len(self.data_set)))
        del data_set
        if self.is_training:
            shuffle(self.data_idx)

    def writeVocab(self, examples):
        words = Counter()  # 定义一个计数器。键是词，值是这个词的频次
        for e in examples:
            words.update(e["words"])
        vocab_file = os.path.join(self.vocab_dir, "vocab.txt")

        if not self.remove_vocab and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.vocab_list = [l.strip() for l in lines]
            self.vocab_size = len(self.vocab_list)
            print("get vocabulary from {}".format(vocab_file))
        else:
            with open(vocab_file, "w", encoding="utf-8") as f:
                for word in words.most_common():
                    f.write(word[0] + '\n')
                    self.vocab_list.append(word[0])
            print("write vocabulary to {}".format(vocab_file))

    def get_vocab_list(self):
        return self.vocab_list

    def __iter__(self):
        return self

    def __next__(self):
        length = len(self.data_idx)
        if self.is_training:
            word_idx = np.zeros(shape=(self.batch_size, self.max_words_length), dtype=np.int32)
            labels = np.zeros(shape=(self.batch_size,), dtype=np.int32)
            id_length = np.zeros(shape=(self.batch_size,), dtype=np.int32)
            for i in range(self.batch_size):
                if self._cur >= length:
                    self._cur = 0
                    shuffle(self.data_idx)
                data = self.data_set[self.data_idx[self._cur]]
                word_idx[i][:len(data["word_ids"])] = np.array(data["word_ids"])
                labels[i] = data["label"]
                id_length[i] = len(data["word_ids"])
                self._cur += 1
            return word_idx, id_length, labels
        else:
            if self._cur >= length:
                self._cur = 0
                raise StopIteration
            batch_size = min(self.batch_size, length - self._cur)
            word_idx = np.zeros(shape=(batch_size, self.max_words_length), dtype=np.int32)
            labels = np.zeros(shape=(batch_size,), dtype=np.int32)
            id_length = np.zeros(shape=(batch_size,), dtype=np.int32)
            for i in range(batch_size):
                data = self.data_set[self._cur]
                word_idx[i][:len(data["word_ids"])] = np.array(data["word_ids"])
                labels[i] = data["label"]
                id_length[i] = len(data["word_ids"])
                self._cur += 1
            return word_idx, id_length, labels

    def get_labels(self):
        all_labels = []
        for i in range(self.num_classes):
            all_labels.append(self.idx_to_textlabel[i])
        return all_labels
