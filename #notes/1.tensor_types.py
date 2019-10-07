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
        a = tf.constant(2)  # Const:0
        b = tf.constant(3)  # Const_1:0
        c = a + b           # Add:0
        print(sess.run(c))


def names():
    """
    This graph will look the same as no_names() except it will have labels
                [ Sum ]
               ↗       ↖
            { A }     { B }

    """
    with tf.Session() as sess:
        a = tf.constant(2, name='A')  # A:0
        b = tf.constant(3, name='B')  # B:0
        c = tf.add(a, b, name='Sum')  # Sum:0
        print(sess.run(c))


def types():
    # Here the constants are assigned a different type and shape
    with tf.Session() as sess:
        s = tf.constant(2.3, name='Scalar', dtype=tf.float32)   # Scalar:0 (dtype=float32)
        m = tf.constant([[1, 2], [3, 4]], name='Matrix')        # Matrix:0 (shape=(2,2), dtype=int32)

        print(s, m)


def variables_old():
    """
    This is the old method of initializing Variables.

    Basic Args for Variable
    tf.Variable(value, name=None)
    """
    with tf.Session() as sess:
        s = tf.Variable(2, name='Scalar')                   # Scalar:0
        m = tf.Variable([[1, 2], [3, 4]], name='Matrix')    # Matrix:0      (shape=(2, 2))
        w = tf.Variable(tf.zeros([784, 10]))                # Variable:0    (shape=(784, 10))

        print(s, m, w)


def variables_new():
    """
    This is the newer recommended way to create a Variable with TensorFlow

    """
    with tf.Session() as sess:
        s = tf.get_variable('Scalar', initializer=tf.constant(2))
        m = tf.get_variable('Matrix', initializer=tf.constant([[1, 2], [3, 4]]))
        # w = tf.


