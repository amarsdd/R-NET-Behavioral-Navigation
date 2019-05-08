"""
Adapted from:
https://github.com/YerevaNN/R-NET-in-Keras

"""
# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer
from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D

from Rnet_layers import *

class RNet(Model):
    def __init__(self, inputs=None, outputs=None, embedding_matrix=None,
                       N=100, M=100, C=25, nB=1, unroll=False,
                       hdim=75, in_dim = 300, word2vec_dim=300, numwords=1000,
                       dropout_rate=0,
                       **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(RNet, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       **kwargs)
            return

        '''Dimensions'''
        B = None
        H = hdim
        W = in_dim

        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')
        VQ_r = SharedWeight(size=(H, H), name='VQ_r')

        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r]

        P_vecs = Input(shape=(N, W), name='P_vecs')
        Q_vecs = Input(shape=(M, 2*W), name='Q_vecs')
        Q_tp = Input(shape=(M, ), name='Q_tp')


        P = P_vecs
        Q = Q_vecs
        Qt = Q_tp
        input_placeholders = [P_vecs, Q_vecs, Q_tp]



        uP = Masking() (P)
        for i in range(nB):
            uP = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll)) (uP)
        uP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uP') (uP)

        embedding_layer = Embedding(numwords,
                                    word2vec_dim,
                                    weights=[embedding_matrix],
                                    input_length=M,
                                    trainable=False)
        Qt = embedding_layer(Qt)
        uQt = Masking() (Qt)
        for i in range(2):
            uQt = Bidirectional(GRU(units= int(H/2),
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll)) (uQt)

        uQ = Masking()(Q)
        uQ = TimeDistributed(Dense(4 * H, activation='relu'))(uQ)
        uQ = TimeDistributed(Dense(H, activation='relu'))(uQ)

        uQ = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, H), name='uQ') (uQ)

        uQ = Concatenate(axis=-1)([uQ, uQt])

        vP = QuestionAttnGRU(units=H,
                             return_sequences=True,
                             unroll=unroll) ([
                                 uP, uQ,
                                 WQ_u, WP_v, WP_u, v, W_g1
                             ])
        vP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, H), name='vP') (vP)

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True,
                                       unroll=unroll)) ([
                                          vP, vP,
                                          WP_v, WPP_v, v, W_g2
                                      ])

        hP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='hP') (hP)

        gP = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               unroll=unroll)) (hP)

        rQ = QuestionPooling() ([uQ, WQ_u, WQ_v, v, VQ_r])
        rQ = Dropout(rate=dropout_rate, name='rQ') (rQ)

        fake_input = GlobalMaxPooling1D() (P)
        fake_input = RepeatVector(n=M, name='fake_input') (fake_input)

        ps = PointerGRU(units=2 * H,
                        return_sequences=True,
                        initial_state_provided=True,
                        name='ps',
                        unroll=unroll) ([
                            fake_input, gP,
                            WP_h, Wa_h, v,
                            rQ
                        ])

        inputs = input_placeholders + shared_weights
        outputs = ps

        super(RNet, self).__init__(inputs=inputs,
                                   outputs=outputs,
                                   **kwargs)
