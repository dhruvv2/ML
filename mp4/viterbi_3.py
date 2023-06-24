"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math

def suffix(word):
    options = ['ation', 'ative', 'eous', 'est', 'ful', 'ial', 'ible', 'ic', 'ion', 'ious', 'ition', 'itive', 'ity', 'ive', 'less', 'ment', 'ness', 'ous', 'tion', 'ing', 'ly', 'ed', 'en', 'er', 'es', 's', 'able', 'y', 'al']
    for suf in options:
        if len(word) >= len(suf):
            if word[-len(suf)] == suf:
                return suf
    return ''
    

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    unique_tags = dict()
    transition_probability = dict()
    emission_probability = {'START':{}}
    emission_probability['START']['START'] = len(train)
    alpha = 0.00001
    words = dict()
    word_counts = {}
    word_token = {}
    
    for sentence in train:
        for i in range(1, len(sentence)):
            if sentence[i] not in word_counts:
                word_counts[sentence[i][0]] = 0
                word_token[sentence[i][0]] = sentence[i][1]
            word_counts[sentence[i][0]] += 1
            if sentence[i][0] not in words:
                words[sentence[i][0]] = 0
            words[sentence[i][0]] += 1
            if sentence[i][1] not in unique_tags:
                unique_tags[sentence[i][1]] = 0
            unique_tags[sentence[i][1]] += 1
            
            if sentence[i-1][1] not in transition_probability:
                transition_probability[sentence[i-1][1]] = dict()
                
            if sentence[i][1] not in emission_probability:
                emission_probability[sentence[i][1]] = dict()
            
            if sentence[i][1] not in transition_probability[sentence[i-1][1]]:
                transition_probability[sentence[i-1][1]][sentence[i][1]] = 0
            
            if sentence[i][0] not in emission_probability[sentence[i][1]]:
                emission_probability[sentence[i][1]][sentence[i][0]] = 0
            
            transition_probability[sentence[i-1][1]][sentence[i][1]] += 1
            emission_probability[sentence[i][1]][sentence[i][0]] += 1
            
    for w in word_counts:
        if word_counts[w] == 1:
            suf = suffix(w)
            if suf not in emission_probability[word_token[w]]:
                emission_probability[word_token[w]][suf] = 0
            emission_probability[word_token[w]][suf] += 1
            
                        
    dics = [transition_probability, emission_probability]
    defaults = []
    for i in range(len(dics)):
        dic = {}
        for key1, d in dics[i].items():
            total = sum(d.values())
            for key2, value in d.items():
                d[key2] = value / total
            dic[key1] = alpha / total
        defaults.append(dic)

    output = []
    unique_tags.pop('END')

    for sentence in test:
        path_tags = []
        path_probs = []

        for i in range(len(sentence)):
            path_tags.append(dict())
            path_probs.append(dict())

        for tag in unique_tags:
            a = math.log(defaults[0][tag])
            if tag in transition_probability['START']:
                a = math.log(transition_probability['START'][tag])
            b = math.log(defaults[1][tag])
            if sentence[i] in emission_probability[tag]:
                b = math.log(emission_probability[tag][sentence[0]])
            path_probs[0][tag] = a + b

        for i in range(1, len(sentence)):
            for tag in unique_tags:
                unknown_tag = 'X'
                neg_infinity = -math.inf

                for final in unique_tags:
                    w = sentence[i]
                    if w not in word_counts:
                        w = suffix(w)
                    
                    a = math.log(defaults[0][final])
                    if tag in transition_probability[final]:
                        a = math.log(transition_probability[final][tag])
                    b = math.log(defaults[1][tag])
                    if w in emission_probability[tag]:
                        b = math.log(emission_probability[tag][w])
                    c = path_probs[i-1][final]
                    
                    if a + b + c > neg_infinity:
                        unknown_tag = final
                        neg_infinity = a + b + c                 
                
                path_tags[i][tag] = unknown_tag
                path_probs[i][tag] = neg_infinity
        
        final = path_tags[-1][max(path_probs[-1], key = path_probs[-1].get)]
        path = [None for _ in range(len(sentence))]
        path[0] = ('START', 'START')    
        path[-1] = ('END', 'END')
        for i in range(len(sentence) - 2):
            path[len(path) - i - 2] = (sentence[len(sentence) - 2 - i], final)
            final = path_tags[len(sentence) - 2 - i][final]
        output.append(path)


    return output