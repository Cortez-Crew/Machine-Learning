"""
My notes for TensorFlow
↖ ↗ ↘ ↙

"""

import tensorflow as tf

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
        { x }   { y }

"""

# Here is an add function
a = 1
b = 2
c = tf.add(a, b, name='Add')
print(c)
# However the add function

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b, name='Add')









