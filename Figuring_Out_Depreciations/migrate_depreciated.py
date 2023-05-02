# https://www.tensorflow.org/guide/migrate/migrating_feature_columns

# Migrate tf.feature_columns to Keras preprocessing layers


import tensorflow as tf
import tensorflow.compat.v1 as tf1
import math


def call_feature_columns(feature_columns, inputs):
    # This is a convenient way to call a `feature_column` outside of an
    # estimator to display its output.
    feature_layer = tf1.keras.layers.DenseFeatures(feature_columns)
    return feature_layer(inputs)


# To use feature columns with an estimator, model inputs are always 
# expected to be a dictionary of tensors:
input_dict = {
    'foo': tf.constant([1]),
    'bar': tf.constant([0]),
    'baz': tf.constant([-1])
}

# # OLD
# # Each feature column needs to be created with a key to index into the 
# # source data. The output of all feature columns is concatenated and
# # used by the estimator model.
# columns = [
#     tf1.feature_column.numeric_column('bar'),
#     tf1.feature_column.numeric_column('baz'),
#     tf1.feature_column.numeric_column('foo'),
# ]
# call_feature_columns(columns, input_dict)


inputs = {
    'foo': tf.keras.Input(shape=()),
    'bar': tf.keras.Input(shape=()),
    'baz': tf.keras.Input(shape=()),
}
# Inputs are typically transformed by preprocessing layers before concatenation.
outputs = tf.keras.layers.Concatenate()(inputs.values())
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model(input_dict)
# print(input_dict)
# # {'foo': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([1])>, 
# # 'bar': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([0])>, 
# # 'baz': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([-1])>
# # }


# # OLD
# # One-hot encoding integer IDs
# # A common feature transformation is one-hot encoding integer inputs of 
# # a known range. Here is an example using feature columns:
# categorical_col = tf1.feature_column.categorical_column_with_identity(
#     'type', num_buckets=3)
# indicator_col = tf1.feature_column.indicator_column(categorical_col)
# call_feature_columns(indicator_col, {'type': [0, 1, 2]})


# Using Keras preprocessing layers, these columns can be replaced by a 
# single tf.keras.layers.CategoryEncoding layer with output_mode set to 
# 'one_hot':

one_hot_layer = tf.keras.layers.CategoryEncoding(
    num_tokens=3, output_mode='one_hot')
one_hot_layer([0, 1, 2])







