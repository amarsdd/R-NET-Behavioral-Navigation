# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU


class WrappedGRU(GRU):

    def __init__(self, initial_state_provided=False, **kwargs):
        kwargs['implementation'] = kwargs.get('implementation', 2)
        assert (kwargs['implementation'] == 2)

        super(WrappedGRU, self).__init__(**kwargs)
        self.input_spec = None
        self.initial_state_provided = initial_state_provided

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if self.initial_state_provided:
            initial_state = inputs[-1:]
            inputs = inputs[:-1]

            initial_state_mask = mask[-1:]
            mask = mask[:-1] if mask is not None else None

        self._non_sequences = inputs[1:]
        inputs = inputs[:1]

        self._mask_non_sequences = []
        if mask is not None:
            self._mask_non_sequences = mask[1:]
            mask = mask[:1]
        self._mask_non_sequences = [mask for mask in self._mask_non_sequences
                                    if mask is not None]

        if self.initial_state_provided:
            assert (len(inputs) == len(initial_state))
            inputs += initial_state

        if len(inputs) == 1:
            inputs = inputs[0]

        if isinstance(mask, list) and len(mask) == 1:
            mask = mask[0]

        return super(WrappedGRU, self).call(inputs, mask, training)

    def get_constants(self, inputs, training=None):
        constants = super(WrappedGRU, self).get_constants(inputs, training=training)
        constants += self._non_sequences
        constants += self._mask_non_sequences
        return constants

    def get_config(self):
        config = {'initial_state_provided': self.initial_state_provided}
        base_config = super(WrappedGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.engine.topology import Layer


class VariationalDropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(VariationalDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            symbolic_shape = K.shape(inputs)
            noise_shape = [shape if shape > 0 else symbolic_shape[axis]
                           for axis, shape in enumerate(self.noise_shape)]
            noise_shape = tuple(noise_shape)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(VariationalDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

# from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax


class PointerGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units // 2
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 6)

        assert (len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert (len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]
        assert (H_ == 2 * H)

        self.input_spec = [None]
        super(PointerGRU, self).build(input_shape=(B, T, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs  # TODO TODO TODO

    def step(self, inputs, states):
        # input
        ha_tm1 = states[0]  # (B, 2H)
        _ = states[1:3]  # ignore internal dropout/masks
        hP, WP_h, Wa_h, v = states[3:7]  # (B, P, 2H)
        hP_mask, = states[7:8]

        WP_h_Dot = K.dot(hP, WP_h)  # (B, P, H)
        Wa_h_Dot = K.dot(K.expand_dims(ha_tm1, axis=1), Wa_h)  # (B, 1, H)

        s_t_hat = K.tanh(WP_h_Dot + Wa_h_Dot)  # (B, P, H)
        s_t = K.dot(s_t_hat, v)  # (B, P, 1)
        s_t = K.batch_flatten(s_t)  # (B, P)
        a_t = softmax(s_t, mask=hP_mask, axis=1)  # (B, P)
        c_t = K.batch_dot(hP, a_t, axes=[1, 1])  # (B, 2H)

        GRU_inputs = c_t
        ha_t, (ha_t_,) = super(PointerGRU, self).step(GRU_inputs, states)

        return a_t, [ha_t]

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 5)

        assert (len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert (len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]

        if self.return_sequences:
            return (B, T, P)
        else:
            return (B, P)

    def compute_mask(self, inputs, mask=None):
        return None  # TODO


# from keras import backend as K

# from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax


class QuestionAttnGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 2)

        assert (len(input_shape[0]) == 3)
        B, P, H_ = input_shape[0]
        assert (H_ == 2 * H)

        assert (len(input_shape[1]) == 3)
        B, Q, H_ = input_shape[1]
        assert (H_ == 2 * H)

        self.input_spec = [None]
        super(QuestionAttnGRU, self).build(input_shape=(B, P, 4 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

    def step(self, inputs, states):
        uP_t = inputs
        vP_tm1 = states[0]
        _ = states[1:3]  # ignore internal dropout/masks
        uQ, WQ_u, WP_v, WP_u, v, W_g1 = states[3:9]
        uQ_mask, = states[9:10]

        WQ_u_Dot = K.dot(uQ, WQ_u)  # WQ_u
        WP_v_Dot = K.dot(K.expand_dims(vP_tm1, axis=1), WP_v)  # WP_v
        WP_u_Dot = K.dot(K.expand_dims(uP_t, axis=1), WP_u)  # WP_u

        s_t_hat = K.tanh(WQ_u_Dot + WP_v_Dot + WP_u_Dot)

        s_t = K.dot(s_t_hat, v)  # v
        s_t = K.batch_flatten(s_t)
        a_t = softmax(s_t, mask=uQ_mask, axis=1)
        c_t = K.batch_dot(a_t, uQ, axes=[1, 1])

        GRU_inputs = K.concatenate([uP_t, c_t])
        g = K.sigmoid(K.dot(GRU_inputs, W_g1))  # W_g1
        GRU_inputs = g * GRU_inputs
        vP_t, s = super(QuestionAttnGRU, self).step(GRU_inputs, states)

        return vP_t, s


from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

from helpers import compute_mask, softmax


class QuestionPooling(Layer):

    def __init__(self, **kwargs):
        super(QuestionPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 5)

        input_shape = input_shape[0]
        B, Q, H = input_shape

        return (B, H)

    def build(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 5)
        input_shape = input_shape[0]

        B, Q, H_ = input_shape
        H = H_ // 2

    def call(self, inputs, mask=None):
        assert (isinstance(inputs, list) and len(inputs) == 5)
        uQ, WQ_u, WQ_v, v, VQ_r = inputs
        uQ_mask = mask[0] if mask is not None else None

        ones = K.ones_like(K.sum(uQ, axis=1, keepdims=True))  # (B, 1, 2H)
        s_hat = K.dot(uQ, WQ_u)
        s_hat += K.dot(ones, K.dot(WQ_v, VQ_r))
        s_hat = K.tanh(s_hat)
        s = K.dot(s_hat, v)
        s = K.batch_flatten(s)

        a = softmax(s, mask=uQ_mask, axis=1)

        rQ = K.batch_dot(uQ, a, axes=[1, 1])

        return rQ

    def compute_mask(self, input, mask=None):
        return None


# from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax


class SelfAttnGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 2)

        assert (len(input_shape[0]) == 3)
        B, P, H_ = input_shape[0]
        assert (H_ == H)

        assert (len(input_shape[1]) == 3)
        B, P_, H_ = input_shape[1]
        assert (P_ == P)
        assert (H_ == H)

        self.input_spec = [None]
        super(SelfAttnGRU, self).build(input_shape=(B, P, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

    def step(self, inputs, states):
        vP_t = inputs
        hP_tm1 = states[0]
        _ = states[1:3]  # ignore internal dropout/masks
        vP, WP_v, WPP_v, v, W_g2 = states[3:8]
        vP_mask, = states[8:]

        WP_v_Dot = K.dot(vP, WP_v)
        WPP_v_Dot = K.dot(K.expand_dims(vP_t, axis=1), WPP_v)

        s_t_hat = K.tanh(WPP_v_Dot + WP_v_Dot)
        s_t = K.dot(s_t_hat, v)
        s_t = K.batch_flatten(s_t)

        a_t = softmax(s_t, mask=vP_mask, axis=1)

        c_t = K.batch_dot(a_t, vP, axes=[1, 1])

        GRU_inputs = K.concatenate([vP_t, c_t])
        g = K.sigmoid(K.dot(GRU_inputs, W_g2))
        GRU_inputs = g * GRU_inputs

        hP_t, s = super(SelfAttnGRU, self).step(GRU_inputs, states)

        return hP_t, s


from keras import initializers
from keras import regularizers

from keras.engine.topology import Node
from keras.layers import Layer, InputLayer


class SharedWeightLayer(InputLayer):
    def __init__(self,
                 size,
                 initializer='glorot_uniform',
                 regularizer=None,
                 name=None,
                 **kwargs):
        self.size = tuple(size)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)

        if not name:
            prefix = 'shared_weight'
            name = prefix + '_' + str(K.get_uid(prefix))

        Layer.__init__(self, name=name, **kwargs)

        with K.name_scope(self.name):
            self.kernel = self.add_weight(shape=self.size,
                                          initializer=self.initializer,
                                          name='kernel',
                                          regularizer=self.regularizer)

        self.trainable = True
        self.built = True
        # self.sparse = sparse

        input_tensor = self.kernel * 1.0

        self.is_placeholder = False
        input_tensor._keras_shape = self.size

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)

        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[self.size],
             output_shapes=[self.size])

    def get_config(self):
        config = {
            'size': self.size,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer)
        }
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))


def SharedWeight(**kwargs):
    input_layer = SharedWeightLayer(**kwargs)

    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


import numpy as np
# from keras import backend as K
from keras.engine import Layer, InputSpec

class Argmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# import numpy as np
# from keras import backend as K
from keras.engine import Layer, InputSpec


class Slice(Layer):
    def __init__(self, indices, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis

        if isinstance(indices, slice):
            self.indices = (indices.start, indices.stop, indices.step)
        else:
            self.indices = indices

        self.slices = [slice(None)] * self.axis

        if isinstance(self.indices, int):
            self.slices.append(self.indices)
        elif isinstance(self.indices, (list, tuple)):
            self.slices.append(slice(*self.indices))
        else:
            raise TypeError("indices must be int or slice")

        super(Slice, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs[self.slices]

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        for i, slice in enumerate(self.slices):
            if i == self.axis:
                continue
            start = slice.start or 0
            stop = slice.stop or input_shape[i]
            step = slice.step or 1
            input_shape[i] = None if stop is None else (stop - start) // step
        del input_shape[self.axis]

        return tuple(input_shape)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return mask
        if self.axis == 1:
            return mask[self.slices]
        else:
            return mask

    def get_config(self):
        config = {'axis': self.axis,
                  'indices': self.indices}
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))