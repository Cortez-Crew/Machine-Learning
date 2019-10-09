"""
My notes for TensorFlow
https://github.com/easy-tensorflow/easy-tensorflow/
↖ ↗ ↘ ↙

"""

import tensorflow.compat.v1 as tf

"""

 ~ A 'Tensor' is a multi-dimensional array of size n

 ~ A 'Graph' represents the flow of the data
 ~ A 'Session' is the actual execution of the 'Graph'


Let's say we want to write code for the function:
        f(x, y) = (x^2 * y) + y + 2


In this case the Tensorflow graph would look like:
  
                    [ Add ]
                    ↗     ↖
               [ Add ]  [ Add ]
                ↗     ↖ ↗     ↖
            [ Add ]  { y }   { 2 }
            ↗     ↖
        { x }   { x }

"""


def add_functions():
    # Here is an add function
    a = 1
    b = 2
    c = tf.add(a, b, name='Add')
    print(c)    # Tensor("Add:0", shape=(), dtype=int32)

    """
    This graph looks like:
    
            [ Add ]
            ↗     ↖
         { x }   { y }
                            * Note how the TensorFlow-names dont match the Python-names when they aren't constants

        Since were creating a TensorFlow graph by using tf.add(),
        we must create a session to actually execute the graph. That's
        why when we print 'c', an empty tensor is returned, since the
        session isn't running.

    """
    sess = tf.Session()
    print(sess.run(c))  # 3
    sess.close()

    # Or the following code is the same. The only difference is the
    # session closes automatically when the block ends

    with tf.Session() as sess:
        print(sess.run(c))  # 3


def multiple_functions():
    """

    Graph of the following function:

                     { x }
                        ↘
       [ Power ] ↖   [ Useless ]
          /|﹨     ﹨     /|﹨
           |        ﹨	  |
           |         ﹨	  |
      [ Multiply ]     [ Add ]
        ↗     ↖        ↗     ↖
     { x }   { y }  { x }   { y }

    """
    # power_op = f(x, y) = (x + y)^(x * y)
    # useless_op = f(x, y) = x * (x + y)
    with tf.Session() as sess:
        x = 2
        y = 3
        add_op = tf.add(x, y, 'Add')
        multiply_op = tf.multiply(x, y, 'Multiply')
        power_op = tf.pow(add_op, multiply_op, 'Power')
        useless_op = tf.multiply(x, add_op, 'Useless')

        power_out, useless_out = sess.run([power_op, useless_op])
        print(power_out, useless_out)

