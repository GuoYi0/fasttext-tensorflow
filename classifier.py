import tensorflow as tf
import config as cfg
import dataset


def get_model():
    text_idx = tf.placeholder(dtype=tf.int32, shape=(cfg.batch_size, cfg.max_words))  # (bs, )
    text_length = tf.placeholder(dtype=tf.int32, shape=(cfg.batch_size,))  # 每个文本向量的长度
    init = tf.random_uniform_initializer(-0.1, 0.1)
    const = tf.constant(0.0, dtype=tf.float32, shape=(1, cfg.word_dimension))  # 没有取到的全部是0，
    text_embedding_w = tf.get_variable("inputs", shape=(cfg.vocab_size + cfg.num_oov_vocab_buckets, cfg.word_dimension),
                                       dtype=tf.float32, initializer=init)
    text_embedding_w = tf.concat([const, text_embedding_w], axis=0)
    text_matrix = tf.nn.embedding_lookup(text_embedding_w, text_idx)  # 文本矩阵，（cfg.batch_size, max_words, word_dimension）
    text_matrix = tf.reduce_sum(text_matrix, axis=1)  # (cfg.batch_size, word_dimension)
    text_embedding = text_matrix / tf.cast(text_length[:, tf.newaxis], tf.float32)  # (cfg.batch_size, word_dimension)
    logits = tf.layers.dense(text_embedding, cfg.num_class, kernel_regularizer=tf.nn.l2_loss, name="fc")
    return text_idx, text_length, logits


def train():
    labels = tf.placeholder(dtype=tf.int32, shape=(cfg.batch_size,))
    text_idx, text_length, logits = get_model()

    model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 1.0e-5 + model_loss
    lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=0.1, trainable=False)
    globel_step = tf.train.get_or_create_global_step()
    opt = tf.train.MomentumOptimizer(lr, momentum=cfg.momentum).minimize(total_loss, global_step=globel_step)
    sess = tf.Session()
    for step in range(1, cfg.train_steps+1):
        if step % (cfg.train_steps//2) == 0:
            sess.run(tf.assign(lr, lr.eval()*0.3))
            print("learning rate at step {} decrease {}".format(step, 0.3))


def main():
    train_data = dataset.dataset(data_dir=cfg.train_dir)



if __name__ == "__main__":
    get_model()
