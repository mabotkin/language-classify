import unidecode
import codecs
import sys

if len(sys.argv) < 2:
	print "Please type a file name."
	sys.exit(0)

for k in range(1,len(sys.argv)):
	FILENAME = sys.argv[k]

	fin = codecs.open(FILENAME,encoding="utf-8").read().splitlines()

	for i in range(len(fin)):
		fin[i] = unidecode.unidecode(fin[i])

	fout = open(FILENAME,"w")
	for i in range(len(fin)):
		fout.write(fin[i] + "\n")
	fout.close()
	print "Removed diacritics from " + FILENAME
