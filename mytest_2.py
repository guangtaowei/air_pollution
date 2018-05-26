import os
import sys
import tensorflow as tf
import numpy as np

path_DBN_1 = os.path.join(
    os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "DeepLearning_tutorials"),
    "models")
sys.path.append(path_DBN_1)
from dbn import DBN


class DBN_1(DBN):
    def pretrain(self, sess, X_train, learning_rate=0.1, k=1):
        print('Starting pretraining...')
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=learning_rate, k=k, persistent=None)
            # 训练
            sess.run(train_ops, feed_dict={self.x: X_train})
            # 计算cost
            val_cost = sess.run(cost, feed_dict={self.x: X_train, })
            print("\tPretraing layer {0} cost: {1}".format(i, val_cost))
        # print(sess.run(self.params))

    def finetuning(self, sess, X_train, Y_train, learning_rate=0.1):
        print("Start finetuning...")
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost,
                                                                                           var_list=self.params)
        # 训练
        sess.run(train_op, feed_dict={self.x: X_train, self.y: Y_train})
        # 计算cost
        val_cost = sess.run(self.cost, feed_dict={self.x: X_train, self.y: Y_train})
        val_acc = sess.run(self.accuracy, feed_dict={self.x: X_train, self.y: Y_train})
        print("\tcost: {0}, validation accuacy: {1}".format(val_cost, val_acc))
        # print(sess.run(self.params))
        # print(sess.run(self.layers[-1].output,feed_dict={self.x: X_train}))

        # print("Prediction:",self.output_layer.y_pred)
        print("  ", sess.run(self.output_layer.y_pred, feed_dict={self.x: X_train}))

    def predict(self, X_test):
        print("Prediction:")
        print("  ", sess.run(self.output_layer.y_pred, feed_dict={self.x: X_test}))



x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y_train = np.array([[3], [5], [7], [9], [11], [13], [15], [17], [19], [21]])

learning_rate = 0.01

# set random_seed
tf.set_random_seed(seed=1111)

dbn = DBN_1(n_in=x_train.shape[1], n_out=y_train.shape[1], hidden_layers_sizes=[2, 2])
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(dbn.params)



for i in range(5):
    print("\ni=", i)
    dbn.pretrain(sess, X_train=x_train, learning_rate=learning_rate)
    dbn.finetuning(sess, X_train=x_train, Y_train=y_train, learning_rate=learning_rate)

x_test = np.array([[11, 12]])
dbn.predict(x_test)
print(x_test.shape, type(x_test))
