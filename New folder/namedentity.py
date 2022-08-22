from random import sample
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
  
# process the text and print Named entities
# tokenization

paths = "mary_rhyme.txt"

with open(paths, "r", encoding="utf8") as f:
    sample_text = f.read()

f.close()

train_text = state_union.raw()

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
# function 
def get_named_entity():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()
    except:
        pass
get_named_entity()