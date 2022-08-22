from matplotlib.pyplot import text
import nltk
import re
import string
text = r"For a moment, nothing happened. Then, after a second or so, nothing continued to happen."
text = text.translate(str.maketrans('', '', string.punctuation))
grammar = ('''NP: {<DT>?<JJ>*<NN>} # NP''')
chunkParser = nltk.RegexpParser(grammar)
tagged = nltk.pos_tag(nltk.word_tokenize(text))
tree = chunkParser.parse(tagged)
tree.draw()