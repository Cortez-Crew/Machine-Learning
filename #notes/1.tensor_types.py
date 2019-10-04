"""
My notes for TensorFlow
https://github.com/easy-tensorflow/easy-tensorflow/
↖ ↗ ↘ ↙

"""

import tensorflow.compat.v1 as tf

"""
    As seen earlier, when we use variables from python in a tensor, 
    their name doesnt transfer to the TensorFloe graph. To fix this 
    we can use a TensorFlow constant.
    
    Args for constant:
    tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
"""


def no_names():
    """
        This graph looks like:
                [ add ]
              ↗       ↖
        { const }   { const_1 }

    """
    with tf.Session() as sess:
        a = tf.constant(2)
        b = tf.constant(3)
        c = a + b
        print(sess.run(c))






