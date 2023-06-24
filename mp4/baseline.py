"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    output = []
    occurences = {}
    tagCount = {}
    for sentence in train:
        for i in range(1, len(sentence) -1): # start at 1 b/c of START , end at len -1 b/c of END
            word, tag = sentence[i][0], sentence[i][1]
            if word in occurences:
                if tag in occurences[word]:
                    occurences[word][tag] +=1
                else:
                    occurences[word][tag] = 1
            else:
                occurences[word] = {}
                occurences[word][tag] = 1
            if tag in tagCount:
                tagCount[tag] +=1
            else:
                tagCount[tag] = 1
                
    for sentence in test:
        sentence_list = []
        sentence_list.append(('START', 'START'))
        for i in range(1, len(sentence)-1):
            word = sentence[i]
            if word in occurences:
                sentence_list.append((word, freqTag(occurences[word])))
            else:
                sentence_list.append((word, freqTag(tagCount)))
        sentence_list.append(('END', 'END'))
        output.append(sentence_list)
    return output

def freqTag(tags):
    return max(tags, key = tags.get)
                      


        