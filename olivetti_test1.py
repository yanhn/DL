# -*- coding:utf-8 -*-
import input_olivetti
import tensorflow as tf
import pdb
import sys

if(len(sys.argv)>1):
  iterNum = int(sys.argv[1])
else:
  iterNum = 1000
mnist = input_olivetti.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder('float',[None,2679])
# pdb.set_trace()
W = tf.Variable(tf.zeros([2679,40]))
b = tf.Variable(tf.zeros([40]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder('float',[None,40])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

MMSE = tf.pow(tf.sub(y,y_),2)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(MMSE)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(iterNum):
  batch_xs,batch_ys = mnist.train.next_batch(40)
  sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# pdb.set_trace()
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels})
sess.close()
