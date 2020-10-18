import re
import sys
from random import random
from math import log
from collections import defaultdict

def preprocess_line(line):
    line = line[:-1] # remove \n
    line = re.sub(r'[^a-zA-Z0-9\s.]','',line) # remove all other characters
    line = re.sub(r'[0-9]', '0', line)  # replace 0-9 to 0
    line = line.lower()
    return line

class Language_Model:
    def __init__(self):
        self.lexicon = [chr(i) for i in range(97, 123)] + [' ', '.', '0'] 
        self.prob = dict()
        self.tri_counts = defaultdict(int)

        # Generator all possible combinations, and deal with the '#' sign
        for ch1 in self.lexicon:
            for ch2 in self.lexicon:
                self.prob[ch1 + ch2] = dict.fromkeys(self.lexicon + ['#'], 0)  # If 2 history are normal characters, they can generator a '#'
        for ch1 in self.lexicon:
            self.prob['#' + ch1] = dict.fromkeys(self.lexicon + ['#'], 0)
        self.prob['##'] = dict.fromkeys(self.lexicon, 0)
                
    
    def train_model(self, train_data_file, alpha = 0.08): 
    
        # This part is used to calculate self.tri_counts
        with open(train_data_file) as f:
            for line in f:
                line = preprocess_line(line)
                line = '##' + line + '#'  # add start sign and end sign
                for j in range(len(line) - 2):
                    trigram = line[j:j+3]
                    self.tri_counts[trigram] += 1

        # calculate probability and smoothing
        for hist in self.prob.keys():
            sum = 0
            for prediction in self.prob[hist].keys():
                sum = sum + self.tri_counts[hist + prediction]
                    #print(hist + prediction + ':' + str(self.tri_counts[hist + prediction]))

            for prediction in self.prob[hist].keys():
                self.prob[hist][prediction] = (self.tri_counts[hist + prediction] + alpha) / (sum + alpha * len(self.prob[hist])) # smooth
        

    def read_model(self, model_file):
        with open(model_file) as f:
            for line in f:
                tri, probability =  line.split('\t')
                self.prob[tri[:2]][tri[2]] = float(probability)


    def print_model(self, output_file):
        with open(output_file, 'w') as f:
            for history in sorted(self.prob.keys()):
                for prediction in sorted(self.prob[history].keys()):
                    f.write(history + prediction + '\t' + str(self.prob[history][prediction])+'\n')

    def generate_from_LM(self):
        history = '##'
        result = ''
        for i in range(300):
            generator = random()
            sum_p = 0 
            for prediction in self.prob[history].keys():
                sum_p  += self.prob[history][prediction]
                if (sum_p > generator): #sum_p > generator means the generator character is c
                    result += prediction
                    if (prediction  == '#'):
                        history = '##'  # begin a new sequence, if last prediction is a end sign
                    else:
                        history = history[1] + prediction 
                    break
        return result

    def calculate_perplexity(self, test):
        n_count = 0
        log_P  = 0
        with open(test) as f:

            for line in f:
                line = preprocess_line(line) + '#'   # add end sign
                history = '##'
                for prediction in line:
                    n_count += 1
                    log_P += log(self.prob[history][prediction], 2)
                    history = history[1] + prediction
        
        perlexity = 2 ** (-1 / n_count * log_P)
        return perlexity


if __name__ == '__main__':
    model = Language_Model()
    model.train_model('training.en')
    model.generate_from_LM()
    model.calculate_perlexity('test')
    model.print_model('out_my.en')
    model.read_model('model-br.en')
    #model.print_model('ok.txt')

    #model.print_model('out_my.en')
    model.generate_from_LM()
        
    model.calculate_perlexity('test')
    #model.print_model('out_my.en')
