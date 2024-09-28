'''Semantic Similarity: starter code

Author: Michael Guerzhoy. Last modified: Nov. 18, 2022.
'''

import math


def norm(vec):
    '''Return the norm of a vector stored as a dictionary, as 
    described in the handout for Project 3.
    '''
    
    sum_of_squares = 0.0  
    for x in vec:
        sum_of_squares += vec[x] * vec[x]
    
    return math.sqrt(sum_of_squares)

def cosine_similarity(vec1, vec2):
    numerator = 0.0
    for x in vec1:
        if x in vec2:
            numerator += vec1[x] * vec2[x]  
    return numerator/(norm(vec1) * norm(vec2))


def build_semantic_descriptors(sentences):
    d = {}
    for sentence in sentences:
        for x in set(sentence):
            x = x.lower()
            #if x in d:
            for y in set(sentence):
                if x in d:
                    if y != x and y in d[x]:
                        d[x][y] += 1
                    elif y != x:
                        d[x][y] = 1
                else:
                    dtemp = {}
                    if y != x and y in dtemp:
                        dtemp[y] += 1
                    elif y != x:
                        dtemp[y] = 1
                    d[x] = dtemp
    return d


def build_semantic_descriptors_from_files(filenames):
    bigtext = ''
    for x in filenames:
        with open(x, 'r', encoding="latin1") as f:
            text = f.read().casefold()
        bigtext += text

    for k in ['?', '!']:
        bigtext = bigtext.replace(k,'.')
    for n in [",", "-", "--", ":", ";", '"',"'", '\n']:
        bigtext = bigtext.replace(n,' ') 

    bigtext = bigtext.split('.')

    for i,x in enumerate(bigtext):
        bigtext[i] = x.split(' ')

    for x in bigtext:
        for n in x:
            while '' in x:
                x.remove('')

    return build_semantic_descriptors(bigtext)


def most_similar_word(word, choices, semantic_descriptors, similarity_fn):
    highest_similarity = 0.0
    similar = choices[0]
    word = word.lower()

    for i in range (len(choices)):
        choices[i] = choices[i].lower()

    if word not in semantic_descriptors:
        return similar

    for x in choices:
        similarity = -1 if x not in semantic_descriptors else similarity_fn(semantic_descriptors[word],semantic_descriptors[x])
        if similarity > highest_similarity and word!=x:
            highest_similarity = similarity
            similar = x
    return similar


def run_similarity_test(filename, semantic_descriptors, similarity_fn):
    filename = open(filename, 'r',encoding='latin1').read().casefold().split('\n')
    tests = []
    for x in filename:
        tests.append(x.split(' '))
    correct = 0.0
    for x in tests:
        if x[1] == most_similar_word(x[0],x[2:],semantic_descriptors,similarity_fn):
            correct += 1
    return correct/len(filename) *100