import tensorflow as tf
import config as cfg
import dataset
import numpy as np
import time
import os

vocab_size = cfg.vocab_size


def get_model():
    text_idx = tf.placeholder(dtype=tf.int32, shape=(None, cfg.max_words))  # (bs, )
    text_length = tf.placeholder(dtype=tf.int32, shape=(None,))  # 每个文本向量的长度
    init = tf.random_uniform_initializer(-0.1, 0.1)
    const = tf.constant(0.0, dtype=tf.float32, shape=(1, cfg.word_dimension))  # 没有取到的全部是0，
    text_embedding_w = tf.get_variable("inputs", shape=(vocab_size, cfg.word_dimension),
                                       dtype=tf.float32, initializer=init)
    print(text_embedding_w.shape, "==================")
    text_embedding_w = tf.concat([const, text_embedding_w], axis=0)
    text_matrix = tf.nn.embedding_lookup(text_embedding_w, text_idx)  # 文本矩阵，（cfg.batch_size, max_words, word_dimension）
    text_matrix = tf.reduce_sum(text_matrix, axis=1)  # (cfg.batch_size, word_dimension)
    text_embedding = text_matrix / tf.cast(text_length[:, tf.newaxis], tf.float32)  # (cfg.batch_size, word_dimension)
    logits = tf.layers.dense(text_embedding, cfg.num_class, kernel_regularizer=tf.nn.l2_loss, name="fc")
    return text_idx, text_length, logits


def evaluate(data):
    text_idx, text_length, logits = get_model()
    prob_tf = tf.nn.softmax(logits, axis=-1)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(cfg.ckpt_path)
    print("testing from {}".format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)
    tp = np.zeros(shape=(cfg.num_class,), dtype=np.int32)
    fp = np.zeros(shape=(cfg.num_class,), dtype=np.int32)
    nums = np.zeros(shape=(cfg.num_class,), dtype=np.int32)
    yes = 0
    for word_idx, id_length, true_labels in data:
        probs = sess.run(prob_tf, feed_dict={text_idx: word_idx, text_length: id_length})
        for prob, true_label in zip(probs, true_labels):  # 一批一批地来
            pred_label = np.argmax(prob)
            nums[true_label] += 1
            if pred_label == true_label:
                tp[pred_label] += 1
                yes += 1
            else:
                fp[pred_label] += 1
    sess.close()

    precision = tp / (tp + fp + 0.001)
    recall = tp / (nums + 0.001)
    f1 = 2 * precision * recall / (precision + recall + 0.001)
    accuracy = yes / nums.sum()
    print(yes, nums.sum())
    with open("result.txt", 'a', encoding='utf-8') as f:
        f.write('=' * 10 + '\n')
        f.write("precision: ")
        for p in precision:
            f.write("%0.3f " % p)
        f.write('\n')
        f.write("recall: ")
        for r in recall:
            f.write("%0.3f " % r)
        f.write('\nf1: ')
        for ff in f1:
            f.write("%0.3f " % ff)
        f.write('\naccuracy: %0.3f' % accuracy)


def get_accuracy(labels, logits):
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    ac = tf.reduce_mean(tf.cast(tf.equal(labels, pred), tf.float32))
    return ac


def run(train_data, test_data):
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))
    text_idx, text_length, logits = get_model()
    model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 1.0e-4 + tf.reduce_mean(model_loss)
    # total_loss = tf.reduce_mean(model_loss)
    lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=0.01, trainable=False)
    globel_step = tf.train.get_or_create_global_step()
    # opt = tf.train.MomentumOptimizer(lr, momentum=cfg.momentum).minimize(total_loss, global_step=globel_step)
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=globel_step)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    ac_tf = get_accuracy(labels, logits)
    for word_idx, id_length, true_labels in train_data:
        if step >= cfg.train_steps:
            saver.save(sess, os.path.join(cfg.ckpt_path, "fasttext.ckpt"), global_step=globel_step)
            break
        if step % (cfg.train_steps // 3) == 0:
            sess.run(tf.assign(lr, sess.run(lr) * 0.2))
            print("learning rate at step {} decrease {}".format(step, 0.3))
        _, tl, ac = sess.run([opt, total_loss, ac_tf], feed_dict={
            text_idx: word_idx, text_length: id_length, labels: true_labels})
        if step % 100 == 0:
            print("step: %04d, ac: %0.3f, loss: %0.4f, lr: %0.4f" % (step, ac, tl, sess.run(lr)))
        if step % 500 == 0:
            saver.save(sess, os.path.join(cfg.ckpt_path, "fasttext.ckpt"), global_step=globel_step)
            if test_data is not None:
                tf.reset_default_graph()
                evaluate(test_data)
        step += 1

    sess.close()


def main():
    global vocab_size
    t1 = time.time()
    print("it begins to prepare training data")
    train_data = dataset.DataSet(data_dir=cfg.train_file, stop_words_file=cfg.stop_words_file,
                                 label_file=cfg.label_file, vocab_dir=cfg.vocab_dir, vocab_list=None,
                                 remove_vocab=False, split=cfg.split, is_trainng=True,
                                 batch_size=cfg.batch_size, max_words_length=cfg.max_words)
    t2 = time.time()
    print("it spend %0.1f to prepare for training data" % (t2 - t1))
    vocab_list = train_data.get_vocab_list()
    # with open(os.path.join(cfg.vocab_dir, "vocab.txt"), "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    # vocab_list = [l.strip() for l in lines]
    #
    vocab_size = len(vocab_list)
    all_labels = train_data.get_labels()
    test_data = dataset.DataSet(data_dir=cfg.test_file, stop_words_file=cfg.stop_words_file,
                                label_file=cfg.label_file, vocab_dir=cfg.vocab_dir, vocab_list=vocab_list,
                                remove_vocab=False, split=cfg.split, is_trainng=False,
                                batch_size=cfg.eval_batch_size, max_words_length=cfg.max_words)
    t1 = time.time()
    print("it spend %0.1f to prepare for testing data" % (t1 - t2))
    f = open("result.txt", "w", encoding='utf-8')
    f.write('       ')
    for label in all_labels:
        f.write(label + '  ')
    f.write('\n')
    f.close()
    print("Now, begin to train")

    run(train_data, test_data=None)
    tf.reset_default_graph()
    evaluate(test_data)


def evaluate_only():
    global vocab_size
    with open(os.path.join(cfg.vocab_dir, "vocab.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    vocab_list = [l.strip() for l in lines]
    vocab_size = len(vocab_list)

    with open(cfg.label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    all_labels = [l.strip() for l in lines]
    t2 = time.time()
    test_data = dataset.DataSet(data_dir=cfg.test_file,
                                stop_words_file=cfg.stop_words_file,
                                label_file=cfg.label_file, vocab_dir=cfg.vocab_dir, vocab_list=vocab_list,
                                remove_vocab=False, split='	', is_trainng=False,
                                batch_size=cfg.eval_batch_size, max_words_length=cfg.max_words)
    t1 = time.time()
    print("it spend %0.1f to prepare for testing data" % (t1 - t2))
    f = open("result.txt", "w", encoding='utf-8')
    f.write('       ')
    for label in all_labels:
        f.write(label + '  ')
    f.write('\n')
    f.close()
    evaluate(test_data)


if __name__ == "__main__":
    main()
    # evaluate_only()
