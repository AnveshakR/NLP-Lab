import nltk
import string

with open("mary_rhyme.txt", "r", encoding='utf-8') as f:
    data = f.readlines()

for sent in data:
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    grammar = ('''NP: {<DT>?<JJ>*<NN>} # NP''')
    chunkparser = nltk.RegexpParser(grammar)
    tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    tree = chunkparser.parse(tagged)
    tree.draw()