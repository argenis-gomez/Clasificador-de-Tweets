import tensorflow_text as tf_text
from model import *

params = {
    'VOCAB_SIZE': 2**13,
    'EMB_DIM': 64,
    'NB_FILTERS': 100,
    'FFN_UNITS': 256,
    'DROPOUT_RATE': 0.4,
    'MAX_LEN': 118
}


def load_model():

    model = DCNN(vocab_size=params['VOCAB_SIZE'],
                 emb_dim=params['EMB_DIM'],
                 nb_filters=params['NB_FILTERS'],
                 FFN_units=params['FFN_UNITS'],
                 dropout_rate=params['DROPOUT_RATE'])

    model.build((None, params['MAX_LEN']))
    model.load_weights('ckpt/model_weights/model')

    return model


def classify(tweet, model):

    tokenizer = tf_text.BertTokenizer('vocab.txt')

    input = tokenizer.tokenize([tweet])
    input = input.merge_dims(-2, -1)
    input = tf.keras.preprocessing.sequence.pad_sequences(input.to_list(),
                                                          padding="post",
                                                          maxlen=params['MAX_LEN'])
    prediction = model(input, training=False).numpy()

    if prediction[0][0] > .5:
        sentiment = 'positivo'
        value = prediction[0][0] * 100
    else:
        sentiment = 'negativo'
        value = (1 - prediction[0][0]) * 100

    return value, sentiment
