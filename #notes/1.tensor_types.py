"""
My notes for TensorFlow
https://github.com/easy-tensorflow/easy-tensorflow/
↖ ↗ ↘ ↙

"""

import tensorflow.compat.v1 as tf

"""
    As seen earlier, when we use variables from python in a tensor, 
    their name doesnt transfer to the TensorFlow graph. To fix this 
    we can use a TensorFlow constant.
    
    Args for constant:
    tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
"""


def no_names():
    """"""
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
    """"""
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
    """"""
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
    """"""
    """
        This is the newer recommended way to create a Variable with TensorFlow

        tf.get_variable(
                        name,
                        shape=None,
                        dtype=None,
                        initializer=None,
                        regularizer=None,
                        trainable=True,
                        collections=None,
                        caching_device=None,
                        partitioner=None,
                        validate_shape=True,
                        use_resource=None,
                        custom_getter=None,
                        constraint=None
                        )
    """
    with tf.Session() as sess:
        s = tf.get_variable('Scalar', initializer=tf.constant(2))
        m = tf.get_variable('Matrix', initializer=tf.constant([[1, 2], [3, 4]]))
        w = tf.get_variable('Weight_Matrix', shape=(784, 10), initializer=tf.zeros_initializer())

        print(s, m, w)


def initializers():
    """"""
    """
        When using tf.get_variable() we need to initialize the value in the session before we can use it
    """
    with tf.Session() as sess:
        a = tf.get_variable(name='A', initializer=tf.constant(2))
        b = tf.get_variable(name='B', initializer=tf.constant(3))
        c = tf.add(a, b, name='Add')

        # Here the global initializer is set
        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        print(sess.run([a, b, c]))


def weights_and_biases():
    """"""
    """
        Variables are commonly used for weights and biases in neural networks.
         ~ Weights are commonly initialized with tf.truncated_normal_initializer(stddev=value)
         ~ Biases are commonly initialized with tf.zeros_initializer()
    """
    with tf.Session() as sess:
        weights = tf.get_variable(name='W', shape=(2, 3), initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable(name='b', shape=3, initializer=tf.zeros_initializer)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        w, b = sess.run([weights, biases])
        print(f'Weights = {w}')
        print(f'Biases = {b}')


def placeholders():
    """"""
    """
        Placeholders are basically empty variables that will have their values added at a later time.
        Examples:
            tf.placeholder(tf.float32, shape=(5))
            tf.placeholder(dtype=tf.float32, shape=None, name=None)
            tf.placeholder(tf.float32, shape=(None, 794), name='input')
            tf.placeholder(tf.float32, shape=(None, 10), name='label')
    """
    with tf.Session() as sess:
        a = tf.constant([5, 5, 5], dtype=tf.float32, name='A')
        b = tf.placeholder(dtype=tf.float32, shape=3, name='B')
        c = tf.add(a, b, name='Add')

        # In order to use the placeholder we now need to assign values to it.
        # Here, use a dictionary
        d = {b: [1, 2, 3]}

        # Then we need to feed it into the session
        print(sess.run(c, feed_dict=d))     # [6. 7. 8.]
        print(a, b, c)

