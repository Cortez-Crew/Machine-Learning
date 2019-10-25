"""
My notes for TensorFlow
https://github.com/easy-tensorflow/easy-tensorflow/
↖ ↗ ↘ ↙

"""

import tensorflow.compat.v1 as tf

"""
    TensorBoard is a suite of visualization tools provided by TensorFlow. These tool help us display our
    graph in a visual and interactive way to help us debug and understand what it is that our model is doing.

    The two main uses for TensorBoard are to:
        ~ Visualize the graph
        ~ Write summaries to visualize the learning

    When our program is 'TensorBoard-Activated', we will have event files. This is what TensorBord will use
    to display information.
    
    In order to view out summary files, we need to navigate to our directory where our graphs are being saved. 
    Once there run the command
    
    tensorboard --logdir='./graphs' --port 6006
    
    It can then be accessed from http://localhost:6006

    
    ** Note that since ever time we run our program there is a new event file being generated. When we 
"""


def tensorboard_basics():
    # Let's clear any data already recognized by TensorFlow so we have a clean slate
    tf.reset_default_graph()

    with tf.Session() as sess:
        # First let's create out graph
        # a = tf.constant(2)
        # b = tf.constant(3)
        # c = tf.add(a, b)
        # But let's make sure to add TensorFlow names, the Python names won't show up in TensorBoard
        a = tf.constant(2, name='a')
        b = tf.constant(3, name='b')
        c = tf.add(a, b, name='addition')

        # Now let's create out summary writer
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        # However if we wanted to create our writer outside of our session we would use
        # writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        print(sess.run(c))


tensorboard_basics()

