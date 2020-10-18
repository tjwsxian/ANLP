from model import Language_Model

# This function is used to choose the best alpha
def choose_alpha():
    best_alpha = -1
    min_p = 10000
    alpha = 0.00001
    while alpha < 1:
    
        train_en = Language_Model()
        train_en.train_model('split_training.en', alpha)
        perplexity = train_en.calculate_perplexity('dev.en')
        if (perplexity < min_p):
            min_p = perplexity
            best_alpha = alpha
        alpha += 0.005
        #print(perplexity, alpha)
    print('best_alpha:',best_alpha)
    return best_alpha

def problem_3():
    train_en = Language_Model()
    train_en.train_model('training.en')
    print(train_en.prob['ng'])
    
def problem_4():

    train_en = Language_Model()
    train_en.train_model('training.en')
    given_en = Language_Model()
    given_en.read_model('model-br.en')
    print('Generator by training:', train_en.generate_from_LM())
    print('Generator by given model:', given_en.generate_from_LM())

def problem_5():
    train_en = Language_Model()
    train_en.train_model('training.en', 0.08)
    print('perplexity calculate by english model:', train_en.calculate_perplexity('test'))
    train_de = Language_Model()
    train_de.train_model('training.de', 0.08)
    print('perplexity calculate by de model:', train_de.calculate_perplexity('test'))
    train_es = Language_Model()
    train_es.train_model('training.es', 0.08)
    print('perplexity calculate by es model:', train_es.calculate_perplexity('test'))

if __name__ == '__main__':
    #choose_alpha()
    problem_3()
    problem_4()
    problem_5()
