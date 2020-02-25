import json
from pathlib import Path
import numpy as np
import pickle
import spacy
nlp_model = spacy.load('en')
from collections import Counter
import itertools

'''
prepare dataset in the following format

X = [instruction1_masked, ... , instructionN_masked]
y = [(ingriedients, instruction1), ... , (ingriedients, instructionN)]
where masked means that all food entities are deleted and replaced with [MASK]
'''

class data_handling:

    X = []
    y = []
    ''' extract recipe instructions and ingredients form 1Mrecipe data and save in format of y'''
    def transform_json(self):
        with open('../data/layer1.json') as file:
            data = json.load(file)
            for recipe in data:
                sep = " "
                ingredients = [d['text'] for d in recipe['ingredients']]
                instructions = sep.join([d['text'] for d in recipe['instructions']])
                self.y.append((ingredients, instructions))
        pickle.dump(self.y, open( "../data/data_y.p", "wb" ))

    '''ingredient words to sort out'''
    def create_vocabulary(self):
        v = []
        with open('../data/train.json') as file1, open('../data/test.json') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            for recipe in data1:
                #ingredients = sum([ingredient.split() for ingredient in recipe['ingredients']], [])
                ingredients = [ingredient for ingredient in recipe['ingredients']]
                v.extend(ingredients)
            for recipe in data2:
                #ingredients = sum([ingredient.split for ingredient in recipe['ingredients']], [])
                ingredients = [ingredient for ingredient in recipe['ingredients']]
                v.extend(ingredients)
        vocabulary = list(set(v))
        with open('../data/vocab.txt', 'w') as f:
            for item in vocabulary:
                f.write("%s\n" % item)
        f.close()
        #pickle.dump(vocabulary, open("../data/vocab.p", "wb"))

    def create_compound_vocabulary(self):
        tokenized_veggies = []
        with open('../data/veggies.txt') as veggies:
            for line in veggies:
                doc = nlp_model(line.strip())
                lemmatized = ''
                for token in doc:
                    lemmatized += f" {token.lemma_}"
                tokenized_veggies.append(lemmatized.strip().lower())

        my_veggies = []
        with open('../data/vocab.txt') as vocab:
            for line in vocab:
                for v in tokenized_veggies:
                    if v in line:
                        lemmatized = ''
                        for token in nlp_model(line.strip()):
                            lemmatized += f" {token.lemma_}"
                        my_veggies.append(lemmatized.strip().lower())

        # my_veggies = list(set(my_veggies))
        with open('../data/my_veggies_list.txt', 'w') as f:
            for item in my_veggies:
                f.write("%s\n" % item)

    def create_trainset_vocabulary(self):
        tokenized_veggies = []
        with open('../data/veggies.txt') as veggies:
            for line in veggies:
                doc = nlp_model(line.strip())
                lemmatized = ''
                for token in doc:
                    lemmatized += f" {token.lemma_}"
                tokenized_veggies.append(lemmatized.strip().lower())

        my_veggies_trainset = []
        y = pickle.load(open("../data/data_y.p", "rb"))
        ingredients = list(list(zip(*y))[0])
        ingredients_merged = list(itertools.chain(*ingredients))
        delete_numbers = lambda x: ''.join([i for i in x if not i.isdigit()]).strip()
        ingredients_merged = list(map(delete_numbers, ingredients_merged))
        words_to_count = (word for word in ingredients_merged)
        c = Counter(words_to_count)
        most_common = c.most_common(500)
        print(most_common)
        pickle.dump(most_common, open("../data/most_common.p", "wb"))

    '''keep only n most common veggies'''
    def filter_vocab(self):
        self.y = pickle.load(open("../data/data_y.p", "rb"))
        with open('../data/my_veggies.txt') as my_veggies:
            my_veggies_dict = {v:0 for v in my_veggies}
            for line in my_veggies_dict.keys():
                c = 0
                for elem in self.y:
                    for ingredient in elem[0]:
                        lemmatized = ''
                        for token in nlp_model(ingredient.strip()):
                            lemmatized += f" {token.lemma_}"
                        lemmatized = lemmatized.strip().lower()
                        if line.strip() in lemmatized:
                            c+=1
                my_veggies_dict[line] = c
            print(my_veggies_dict)
            pickle.dump(my_veggies_dict, open( "../data/veggies_counted.p", "wb" ))

if __name__ == '__main__':
    dh = data_handling()
    dh.create_trainset_vocabulary()