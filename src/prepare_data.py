import json
from pathlib import Path
import numpy as np
import pickle
import spacy
nlp_model = spacy.load('en')
from collections import Counter
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import random


'''
prepare dataset in the following format

X = [instruction1_masked, ... , instructionN_masked]
y = [(ingriedients, instruction1), ... , (ingriedients, instructionN)]
where masked means that all food entities are deleted and replaced with [MASK]
'''

class data_handling:

    X = []
    y = []
    '''
    extract recipe instructions and ingredients form 1Mrecipe data and save in format of y
    '''
    def transform_json(self):
        with open('../data/layer1.json') as file:
            data = json.load(file)
            for recipe in data:
                sep = " "
                ingredients = [d['text'] for d in recipe['ingredients']]
                instructions = sep.join([d['text'] for d in recipe['instructions']])
                self.y.append((ingredients, instructions))
        pickle.dump(self.y, open( "../data/data_y.p", "wb" ))

    '''
    ingredient words to sort out
    '''
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

    '''
    set of all food stuffs in ingredients dataset
    '''
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

        my_veggies = list(set(my_veggies))
        with open('../data/my_veggies.txt', 'w') as f:
            for item in my_veggies:
                f.write("%s\n" % item)

    '''
    create vocubulary of most common veggie words in instruction
    for computational reasons just use subset of 1500 recipes to base decision on
    '''
    def create_instruction_vocabulary(self):
        tokenized_veggies = []
        with open('../data/veggies.txt') as veggies:
            for line in veggies:
                doc = nlp_model(line.strip())
                lemmatized = ''
                for token in doc:
                    lemmatized += f" {token.lemma_}"
                tokenized_veggies.append(lemmatized.strip().lower())

        y = pickle.load(open("../data/data_y.p", "rb"))
        instructions = list(list(zip(*y))[1])
        random.shuffle(instructions)
        instructions = instructions[:1500]

        instructions_merged = ' '.join(instructions)
        instructions_merged = ''.join([i for i in instructions_merged if not i.isdigit()]).strip()

        #  lemmatize and only keep veggies
        doc = nlp_model(instructions_merged)
        filtered_sentence = []
        for token in doc:
            lemmatized = token.lemma_
            if lemmatized in tokenized_veggies:
                filtered_sentence.append(lemmatized)

        words_to_count = (word for word in filtered_sentence)
        c = Counter(words_to_count)
        most_common = c.most_common(100)
        with open('../data/most_common.txt', 'w') as f:
            for w in most_common:
                print(w)
                f.write(f"{w[0]}\n")
        f.close()


if __name__ == '__main__':
    dh = data_handling()
    dh.create_instruction_vocabulary()