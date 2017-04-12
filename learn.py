# pylint: disable=I0011,C0103
""" Learn Tensorflow """
import tensorflow as tf

def main():
    """ Main """
    session = tf.Session()

    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    linear_model = W * x + b

    squared_detas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_detas)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    session.run(init)

    for _ in range(1000):
        session.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

    print(session.run([W, b]))

if __name__ == "__main__":
    main()
