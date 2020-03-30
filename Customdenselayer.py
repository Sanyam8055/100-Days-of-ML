from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.SNNDense')  
class SNNDense(tf.keras.layers.Layer):
  def __init__(self,
		            units,
                rate=1e-2,
       	      	activation=selu(input),
	            	kernel_initializer='lecun_uniform', 
	        **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(SNNDense, self).__init__()
    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
  input_shape = tensor_shape.TensorShape(input_shape)
  last_dim = tensor_shape.dimension_value(input_shape[-1])
  self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
  self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)

  def call(self, input):
    self.add_loss(self.rate * tf.reduce_sum(inputs))
    rank = inputs.shape.rank
    if rank is not None and rank > 2:
	  outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
     if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernel)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    if self.activation is not None:
      return self.activation(outputs) 
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
    }
    base_config = super(SNNDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
