from keras.models import Sequential
from keras.layers import Dense
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

random.seed(0)

NUM_LANGS = len(LANGS)
DICTS = {}
TOT = []

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
		TOT.append((fin[j],i))
	DICTS[i] = fin
random.shuffle(TOT)
X = []
Y = []
for i in TOT:
	X.append(i[0])
	Y.append(i[1])
X = np.array(X)
Y = np.array(Y)

#generate net
model = Sequential()
model.add(Dense(MAX_WORD_LENGTH, input_dim=MAX_WORD_LENGTH, init='uniform', activation='relu'))
model.add(Dense(MAX_WORD_LENGTH+1, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y, nb_epoch=ITERATIONS, batch_size=BATCH_SIZE)

scores = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

toc = time()
print "Time Elapsed: " + str(toc-tic) + " seconds."
