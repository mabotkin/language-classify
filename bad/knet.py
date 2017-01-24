from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import os
import random
from time import time

# global constants
MAX_WORD_LENGTH = 15
LANGS = ["spanish","french"]
ITERATIONS = 100
BATCH_SIZE = 32
NUM_TOP = 3
DUMP_FILE_NAME = "model.h5"

VERBOSE = True

random.seed(0)

NUM_LANGS = len(LANGS)
DICTS = {}
TOT = []

def convWord(word):
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
	return tmp

regen = True
if os.path.isfile("model.h5"):
	print "Use existing data? [y/n]"
	inp = raw_input()
	if inp != "n":
		regen = False

if regen:
	tic = time()
	# read in data
	for i in range(len(LANGS)):
		fin = open("data/" + LANGS[i] + ".txt").read().splitlines()
		for j in range(len(fin)):
			fin[j] = np.array(convWord(fin[j]))
			TOT.append((fin[j],i))
		DICTS[i] = fin
	random.shuffle(TOT)
	X = []
	Y = []
	for i in TOT:
		X.append(i[0])
		correct = [0.0 for j in range(NUM_LANGS)]
		correct[i[1]] = 1.0
		Y.append(np.array(correct))
	X = np.array(X)
	Y = np.array(Y)

	#generate net
	model = Sequential()
	model.add(Dense(MAX_WORD_LENGTH, input_dim=MAX_WORD_LENGTH, init='uniform', activation='relu'))
	model.add(Dense(MAX_WORD_LENGTH+1, init='uniform', activation='relu'))
	#model.add(Dense(MAX_WORD_LENGTH, init='uniform', activation='relu'))
	#model.add(Dense(MAX_WORD_LENGTH, init='uniform', activation='relu'))
	model.add(Dense(NUM_LANGS, init='uniform', activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X,Y, nb_epoch=ITERATIONS, batch_size=BATCH_SIZE,verbose=VERBOSE)

	scores = model.evaluate(X,Y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	toc = time()
	print "Time Elapsed: " + str(toc-tic) + " seconds."
	model.save("model.h5")
else:
	model = load_model("model.h5")

while True:
	print "Enter word:"
	word = raw_input()
	out = model.predict(np.array(convWord(word)).reshape((1,MAX_WORD_LENGTH)))
	out = out[0]
	tmp = []
	for i in range(len(out)):
		tmp.append((1-out[i],LANGS[i]))
	out = sorted(tmp,reverse=True)
	for i in range(min(NUM_TOP,NUM_LANGS)):
		print out[i][1] + ": " + str(100*out[i][0]) + "%"
