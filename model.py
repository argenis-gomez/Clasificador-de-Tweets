import tensorflow as tf
from tensorflow.keras import layers


class DCNN(tf.keras.Model):

    def __init__(self, vocab_size, emb_dim=128, nb_filters=50, FFN_units=512, dropout_rate=0.1, name="dcnn"):
        super(DCNN, self).__init__(name=name)
        self.embedding = layers.Embedding(vocab_size, emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding="valid", activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding="valid", activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding="valid", activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        self.dense = layers.Dense(units=FFN_units, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(2 * dropout_rate))
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output
