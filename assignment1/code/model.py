import re
import sys
from random import random
from math import log
from collections import defaultdict
import numpy as np

def preprocess_line(line):
	l=[]
	for c in line:
		if(c>='0' and c<='9'):
			l.append('0')
		elif(c>='a' and c<='z'):
			l.append(c)
		elif(c>='A' and c<='Z'):
			l.append(c.lower())
		elif(c==' ' or c=='.'):
			l.append(c)
	line="".join(l)
	return line

class Language_Model:
	def __init__(self):
		self.lexicon = [chr(i) for i in range(97, 123)] + [' ', '.', '0']

		self.prob = dict()

		# Generator all possible combinations, and deal with the '#' sign
		for ch1 in self.lexicon:
			for ch2 in self.lexicon:
				self.prob[ch1 + ch2] = dict.fromkeys(self.lexicon + ['#'], 0.)
				# If 2 history are normal character, they can let a '#'

		for ch1 in self.lexicon:
			self.prob['#' + ch1] = dict.fromkeys(self.lexicon+['#'], 0.)
		self.prob['##'] = dict.fromkeys(self.lexicon, 0.)

	def train_model(self, train_data_file):
		#add-alpha smoothing
		alpha=0.5
		v=30# the number of character types in English
		#calculate the count of each trigram
		print("using the training data:", train_data_file.split('/')[-1])
		with open(train_data_file) as f:
			for line in f:
				if(len(line)==0):
					continue
				line=preprocess_line(line)
				#process the first one and two character
				if(len(line)<=1):
					self.prob['##'][line[0]]+=1
					self.prob['#'+line[0]]['#']+=1
					# input()
					# self.prob['#'+line[0]]['#']+=1# need add the format like #*#?
				else:
					self.prob['##'][line[0]]+=1
					self.prob['#'+line[0]][line[1]]+=1
					for j in range(len(line) - 2):
						trigram = line[j:j + 3]
						self.prob[trigram[0:2]][trigram[2]]+=1
					# process the last two character
					self.prob[line[-2:]]['#']+=1
		#calculate the probabilty of each trigram

		print(self.prob["ar"]["."])

		for k in self.prob.keys():
			cHistory=np.sum(list(self.prob[k].values()))
			v=len(self.prob[k])
			for k2 in self.prob[k]:
				self.prob[k][k2]=(self.prob[k][k2]+alpha)/(cHistory+v*alpha)
		print("hey, man, finish training !!\nnow you can write your language model into your file")
		return
		
	def read_model(self, model_file):
		#read the model file into the self.prob
		with open(model_file) as f:
			for line in f:
				if(line==""):
					print("read the last row")
					continue
				trigram=line[0:3]
				p=float(line[4:])
				self.prob[trigram[0:2]][trigram[2]]=p
		print("the model file has been read into the self.prob")
		return

	def print_model(self, output_file):
		with open(output_file,"w") as f:
			for k1 in self.prob.keys():
				for k2 in self.prob[k1]:
					trigram=k1+k2
					writing=trigram+'\t'+format(self.prob[k1][k2], '.5e')+"\n"
					f.write(writing)
		print("the language model has been writen in ./",output_file)
		return

	def generate_from_LM(self):
		randomSequence = "##"
		history = "##"
		#choose first two character:
		i=0
		while (i<298):
			p = np.random.rand()
			a = 0
			for k in self.prob[history].keys():
				if(p>=a and p<a+self.prob[history][k]):
					if k!='#': #avoid including '#' in the sequence, necessary?
						randomSequence += k
						history = randomSequence[-2:]
						i += 1;
						break
					else:
						randomSequence += k
						history = "##"
						i = i+1
						break
				else:
					a = a + self.prob[history][k]
		return randomSequence

	def calculate_perlexity(self, test):
		with open(test) as f:
			Perplexitysum=0
			num=0
			for l in f:
				l=preprocess_line(l)
				# num+=1
				n=len(l)
				l="##"+l+"#"
				p=0
				for i in range(n+1):
					num+=1
					p=p+np.log2(self.prob[l[i:i+2]][l[i+2]])
				#p=-p/(n+1)
				Perplexitysum+=p
			perlexity=pow(2,-Perplexitysum/num)
		return perlexity

lm=Language_Model()


# task3
def task3():
	lm = Language_Model()
	trainingdata="/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/training.en"
	lm.train_model(trainingdata)
	lm.print_model("LMfrom"+trainingdata.split("/")[-1])

	lm = Language_Model()
	trainingdata="/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/training.es"
	lm.train_model(trainingdata)
	lm.print_model("LMfrom"+trainingdata.split("/")[-1])

	lm = Language_Model()
	trainingdata="/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/training.de"
	lm.train_model(trainingdata)
	lm.print_model("LMfrom"+trainingdata.split("/")[-1])


# task4
def task4():
	lm = Language_Model()
	lm.read_model("LMfromtraining.en")
	# lm.read_model("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/model-br.en")
	sequence=lm.generate_from_LM()
	print("the random sequence:")
	print(sequence)
	print("len:",len(sequence))

#task5
def task5():
	lm = Language_Model()
	lm.read_model("LMfromtraining.en")
	perplexity=lm.calculate_perlexity("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/test")
	print("perplexity from LMfromtraining.en:")
	print(perplexity)

	lm.read_model("LMfromtraining.es")
	perplexity=lm.calculate_perlexity("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/test")
	print("perplexity from LMfromtraining.es:")
	print(perplexity)

	lm.read_model("LMfromtraining.de")
	perplexity=lm.calculate_perlexity("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/test")
	print("perplexity from LMfromtraining.de:")
	print(perplexity)

	lm.read_model("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/model-br.en")
	perplexity=lm.calculate_perlexity("/Users/hzh/Documents/Msc-AI-content/sem1/ANLP/assignment1/assignment1-data/test")
	print("perplexity from model-br.en:")
	print(perplexity)


task3()
# task4()
task5()

