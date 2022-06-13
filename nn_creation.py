from keras.layers import LSTM, Input, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import  RepeatVector, Permute
from keras.layers import Multiply, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import keras.backend as K


def create_network(n_notes, n_durations, embed_size=100, rnn_units=256, use_attention=False):
    """ create the structure of the neural network """

    notes_in = Input(shape=(None,))
    durations_in = Input(shape=(None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([x1, x2])

    x = LSTM(rnn_units, return_sequences=True)(x)
    # x = Dropout(0.2)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)
        # x = Dropout(0.2)(x)

        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)

    else:
        c = LSTM(rnn_units)(x)
        # c = Dropout(0.2)(c)

    notes_out = Dense(n_notes, activation='softmax', name='pitch')(c)
    durations_out = Dense(n_durations, activation='softmax', name='duration')(c)

    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    opti = RMSprop(learning_rate=0.001)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model
