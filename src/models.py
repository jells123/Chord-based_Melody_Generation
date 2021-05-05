
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Input, Concatenate
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Add
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from keras.layers import Layer
import keras.backend as K

from numpy.random import seed
import random as rn
import sys

def set_seed(sd=123):
    print("SEED:", sd)
    print("python {}".format(sys.version))
    print("keras version {}".format(keras.__version__))
    print("tensorflow version {}".format(tf.__version__))
    seed(sd)
    rn.seed(sd)
    tf.random.set_seed(sd)
    
set_seed(13111996)

def get_stateful_lstm_multi_output(batch_input_shape, num_pitches, num_durations, lstm_size, lr=1e-3, loss_weights=[1.0, 1.0], chords=False):
    lstm_f = LSTM(lstm_size, stateful=True, return_sequences=False, name="lstm_all")
    pitch_f = Dense(num_pitches, activation='softmax', name=("pitch_out" if not chords else "chord_out"))
    duration_f = Dense(num_durations, activation='softmax', name="duration_out")

    model_input = Input(batch_shape=batch_input_shape, name='m_input')
    lstm_out = lstm_f(model_input)

    pitch_out = pitch_f(lstm_out)
    duration_out = duration_f(lstm_out)

    model = Model(inputs=[model_input], outputs=[pitch_out, duration_out])
    opt = keras.optimizers.Adam(lr)
    
    model.compile(
        optimizer=opt,
        loss={
            ("pitch_out" if not chords else "chord_out"): keras.losses.CategoricalCrossentropy(from_logits=True),
            "duration_out": keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=loss_weights,
    )
    
    model_name = f"LSTM-{lstm_size}"
    return model, opt, model_name

def get_stateless_lstm_multi_output(input_shape, num_pitches, num_durations, lstm_size, lr=1e-3, loss_weights=[1.0, 1.0], chords=False):
    lstm_f = LSTM(lstm_size, stateful=False, return_sequences=False, name="lstm_all")
    pitch_f = Dense(num_pitches, activation='softmax', name=("pitch_out" if not chords else "chord_out"))
    duration_f = Dense(num_durations, activation='softmax', name="duration_out")

    model_input = Input(batch_shape=batch_input_shape, name='m_input')
    lstm_out = lstm_f(model_input)

    pitch_out = pitch_f(lstm_out)
    duration_out = duration_f(lstm_out)

    model = Model(inputs=[model_input], outputs=[pitch_out, duration_out])
    opt = keras.optimizers.Adam(lr)
    
    model.compile(
        optimizer=opt,
        loss={
            ("pitch_out" if not chords else "chord_out"): keras.losses.CategoricalCrossentropy(from_logits=True),
            "duration_out": keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=loss_weights,
    )
    
    model_name = f"LSTM-{lstm_size}"
    return model, opt, model_name

def get_stateful_lstm_multi_output_deeper_NEW(batch_input_shape, num_pitches, num_durations, lstm_size1, lstm_size2, lr=1e-3, loss_weights=[1.0, 1.0], chords=False):
    lstm_f = LSTM(lstm_size1, stateful=True, return_sequences=True, return_state=True, name="lstm_first")
    
    lstm_next_f = LSTM(lstm_size1, stateful=True, name="lstm_second")
    pitch_f = Dense(num_pitches, activation='softmax', name=("pitch_out" if not chords else "chord_out"))
    duration_f = Dense(num_durations, activation='softmax', name="duration_out")

    model_input = Input(batch_shape=batch_input_shape, name='m_input')
    lstm_out, lstm_state_h, lstm_state_c = lstm_f(model_input)
    lstm_out_next = lstm_next_f(lstm_out)

    first_second_lstm = Add(name="states_sum")([lstm_state_h, lstm_out_next])
    pitch_out = pitch_f(first_second_lstm)
    duration_out = duration_f(first_second_lstm)

    model = Model(inputs=[model_input], outputs=[pitch_out, duration_out])
    opt = keras.optimizers.Adam(lr)
    
    model.compile(
        optimizer=opt,
        loss={
            ("pitch_out" if not chords else "chord_out"): keras.losses.CategoricalCrossentropy(from_logits=True),
            "duration_out": keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=loss_weights,
    )
    
    model_name = f"LSTM-NEW-{lstm_size1}-{lstm_size1}"
    return model, opt, model_name

def get_stateful_lstm_multi_output_deeper_FIX(batch_input_shape, num_pitches, num_durations, lstm_size1, lstm_size2, lr=1e-3, loss_weights=[1.0, 1.0], chords=False):
    model_input = Input(batch_shape=batch_input_shape, name='m_input')
    
    lstm_f = LSTM(lstm_size1, stateful=True, return_sequences=True, return_state=True, name="lstm_first")
    lstm_out, lstm_state_h, lstm_state_c = lstm_f(model_input)

    lstm_next_f_pitch = LSTM(lstm_size1, stateful=True, name="lstm_second_pitch")
    lstm_next_f_duration = LSTM(lstm_size1, stateful=True, name="lstm_second_duration")
    
    lstm_out_next_pitch = lstm_next_f_pitch(lstm_out)
    lstm_out_next_duration = lstm_next_f_duration(lstm_out)

    first_second_lstm_pitch = Add(name="states_sum_pitch")([lstm_state_h, lstm_out_next_pitch])
    first_second_lstm_duration = Add(name="states_sum_duration")([lstm_state_h, lstm_out_next_duration])

    pitch_f = Dense(num_pitches, activation='softmax', name=("pitch_out" if not chords else "chord_out"))
    pitch_out = pitch_f(first_second_lstm_pitch)

    duration_f = Dense(num_durations, activation='softmax', name="duration_out")
    duration_out = duration_f(first_second_lstm_duration)

    model = Model(inputs=[model_input], outputs=[pitch_out, duration_out])
    opt = keras.optimizers.Adam(lr)
    
    model.compile(
        optimizer=opt,
        loss={
            ("pitch_out" if not chords else "chord_out"): keras.losses.CategoricalCrossentropy(from_logits=True),
            "duration_out": keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=loss_weights,
    )
    
    model_name = f"LSTM-NEW-{lstm_size1}-{lstm_size1}"
    return model, opt, model_name


class attention(Layer):
    # https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
    def __init__(self,**kwargs):
            super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
            self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
            self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")                
            super(attention, self).build(input_shape)

    def call(self,x):
            et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
            at=K.softmax(et)
            at=K.expand_dims(at,axis=-1)
            output=x*at
            return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
            return (input_shape[0],input_shape[-1])

    def get_config(self):
            return super(attention,self).get_config()

def get_model_stateless(X_shape, y_shape, mode, lstm_size=256, drop=0.2, lr=1e-5):
    assert mode in ["dummy", "embeddings"]
    model = Sequential()
    model.add(LSTM(lstm_size, input_shape=(X_shape[1], X_shape[2])))
    if drop:
        model.add(Dropout(drop))
    model.add(Dense(y_shape[1], activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model, f"model_{mode}_{lstm_size}"

def get_model_stateful(X_shape, y_shape, mode, lstm_size=128, drop=0.2, lr=1e-5):
    assert mode in ["dummy", "embeddings"]
    model = Sequential()
    model.add(LSTM(lstm_size, batch_input_shape=(1, 1, X_shape[1]), stateful=True))
    if drop:
        model.add(Dropout(drop))
    model.add(Dense(y_shape[1], activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=lr)
    # opt = keras.optimizers.RMSprop()

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model, f"model_{mode}_{lstm_size}_stateful"

def get_model_giga_stateful(X_shape, y_shape, mode, lstm_sizes=[512, 512], drop=0.15, lr=1e-3):
    assert mode in ["dummy", "embeddings"]
    model = Sequential()
    model.add(LSTM(lstm_sizes[0], batch_input_shape=(1, 1, X_shape[1]), stateful=True, return_sequences=True))
    if drop:
        model.add(Dropout(drop))
    for ls in lstm_sizes[1:-1]:
        model.add(LSTM(ls, return_sequences=True))
        if drop:
            model.add(Dropout(drop))
    model.add(LSTM(lstm_sizes[-1]))
    model.add(Dense(y_shape[1], activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=lr)
    # opt = keras.optimizers.RMSprop()

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    
    return model, f"gigamodel_{mode}_N+CH_{'-'.join(map(str, lstm_sizes))}_stateful"

def get_attention_model_stateless(input_shape, vocab_size, lstm_size=512, droprate=0.15):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        LSTM(
            lstm_size,
            return_sequences=True)
    )
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(droprate))

    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(Dropout(droprate))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def TODO_ATTENTION():
    BATCH_SIZE = 1
    LSTM_SIZE = 32
    DROPRATE = 0.05
    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
    VOCAB_SIZE = y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(
        LSTM(
            LSTM_SIZE,
    #         batch_input_shape=(1, 1, X_train.shape[1]),
    #         stateful=True,
            return_sequences=True
        )
    )
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(DROPRATE))

    model.add(LSTM(LSTM_SIZE, return_sequences=False))
    model.add(Dropout(DROPRATE))

    # model.add(Flatten())
    # model.add(GlobalMaxPooling1D())
    model.add(Dense(VOCAB_SIZE))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.summary()