vocab_size = 200  # 没意义的初始化值
word_dimension = 10  # 词向量的长度设定为10
max_words = 1000  # 假设每个文本最多有1000个词
num_class = 10  # 10个类别
batch_size = 256  # 一次训练多少条数据。一般来说，在内存允许范围内尽量大点。
train_steps = 10000  # 训练多少步
split = '\t'
stop_words_file = "E:\\zsm\\ads_cls\\data\\stopwords\\stopwords.txt"  # 停用词集
vocab_dir = "E:\\zsm\\ads_cls\\data\\vocab"  # 词汇表
eval_batch_size = 1  # 测试时候每次多少条数据
ckpt_path = "E:\\zsm\\ads_cls\\ckpt"  # 模型训练的时候保存地址
# =====================================================================================
train_file = "E:\zsm\\ads_cls\\data\\dataset\\train.txt"  # 训练集
test_file = "E:\\zsm\\ads_cls\\data\\dataset\\test.txt"  # 测试集
label_file = "E:\\zsm\\ads_cls\\data\\dataset\\class_zh.txt"  # 标签集
