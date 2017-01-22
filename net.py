import tensorflow as tf
import numpy as np
import random
from time import time

# global constants
MAX_WORD_LENGTH = 15
LANGS = ["spanish","french"]
TRAIN_RATE = 0.7
ITERATIONS = 1000
BATCH_SIZE = 100
STDDEV = 0.1

NUM_LANGS = len(LANGS)
DICTS = {}

tic = time()

# read in data
for i in range(len(LANGS)):
	fin = open("data/" + LANGS[i] + ".txt").read().splitlines()
	for j in range(len(fin)):
		tmp = []
		for k in fin[j]:
			tmp.append(float(ord(k)))
		if len(tmp) > MAX_WORD_LENGTH:
			# truncate if too long
			tmp = tmp[0:MAX_WORD_LENGTH]
		else:
			for n in range(len(tmp),MAX_WORD_LENGTH):
				# pad if too short
				tmp.append(0)
		fin[j] = np.array(tmp)
	DICTS[i] = fin

def batch(lang, num):
	return random.sample(DICTS[lang],num)

# define variables
x = tf.placeholder(tf.float32,[None,MAX_WORD_LENGTH])
#W = tf.Variable(tf.zeros([MAX_WORD_LENGTH,NUM_LANGS]))
W = tf.Variable(tf.truncated_normal([MAX_WORD_LENGTH,NUM_LANGS],stddev=STDDEV), trainable=True)
#b = tf.Variable(tf.zeros([NUM_LANGS]))
b = tf.Variable(tf.truncated_normal([NUM_LANGS],stddev=STDDEV), trainable=True)

# define net
y = tf.nn.softmax(tf.matmul(x, W) + b)
#y = (tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, NUM_LANGS])

# define error + training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(TRAIN_RATE).minimize(cross_entropy)

# actually do the net
sess = tf.InteractiveSession()
init = tf.global_variables_initializer().run()

for k in range(ITERATIONS):
	together = []
	for i in range(NUM_LANGS):
		bat = batch(i,BATCH_SIZE)
		for j in bat:
			together.append((j,i))
	random.shuffle(together)
	tog_x = []
	tog_y = []
	for i in together:
		tog_x.append(i[0])
		tmp = [0 for j in range(NUM_LANGS)]
		tmp[i[1]] = 1.0
		tog_y.append(tmp)
	tog_x = np.matrix(tog_x)
	tog_y = np.matrix(tog_y)
	print tog_x.shape
	print tog_y.shape
	sess.run(train_step, feed_dict={x: tog_x, y_: tog_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ans = y#tf.argmax(y,1)

together = []
for i in range(NUM_LANGS):
	for j in DICTS[i]:
		together.append((j,i))
random.shuffle(together)
tog_x = []
tog_y = []
for i in together:
	tog_x.append(i[0])
	tmp = [0 for j in range(NUM_LANGS)]
	tmp[i[1]] = 1.0
	tog_y.append(tmp)
tog_x = np.matrix(tog_x)
tog_y = np.matrix(tog_y)

print "Accuracy: " + str(sess.run(accuracy, feed_dict={x: tog_x, y_: tog_y}))
toc = time()
print "Time Elapsed: " + str(toc-tic) + " seconds"

while True:
	print "Enter Word: "
	word = raw_input()
	tmp = []
	for k in word:
		tmp.append(float(ord(k)))
	if len(tmp) > MAX_WORD_LENGTH:
		# truncate if too long
		tmp = tmp[0:MAX_WORD_LENGTH]
	else:
		for n in range(len(tmp),MAX_WORD_LENGTH):
			# pad if too short
			tmp.append(0)
	tmp = np.matrix(tmp)
	#print LANGS[sess.run(ans,feed_dict={x: tmp})[0]]
	print sess.run(ans,feed_dict={x: tmp})
