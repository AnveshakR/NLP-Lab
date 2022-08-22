from collections import Counter
text = "It is necessary for any Data Scientist to understand Natural Language Processing"
text = text.lower()
the_count = Counter(text)
print(the_count) 

import nltk
from collections import Counter
text = "It is necessary for any Data Scientist to understand Natural Language Processing"
text = text.lower()
tokens = nltk.word_tokenize(text)
pos = nltk.pos_tag(tokens)
the_count = Counter(tag for _, tag in pos)
print(the_count) 


from nltk.corpus import gutenberg
text = gutenberg.words('melville-moby_dick.txt')
print(len(text))
print(text[:100])

fdistribution = nltk.FreqDist(text)

fdistribution.plot(50, cumulative=True)


text = "Anna Karenina by Leo Tolstoy. So, we've to learn History, French, ,Germann and the like! Yay!"

#Sentence tokenization
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
print(len(sentences), 'sentences:', sentences)

#Word tokenization
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
print(len(words), 'words:', words)


#Filter out stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(len(stop_words), "stopwords:", stop_words)
words = [word for word in words if word not in stop_words]
print(len(words), "without stopwords:", words)


#Tokenize the text first, then filter out the stopwords:
text = "Anna Karenina by Leo Tolstoy. So, we've to learn History, French, ,Germann and the like! Yay!"

from nltk.tokenize import word_tokenize
words = word_tokenize(text)

print(len(words), "in original text:", words)

words = [word for word in words if word not in stop_words]
print(len(words), "without stopwords:", words)

import string
punctuations = list(string.punctuation)
print(punctuations)
words = [word for word in words if word not in punctuations]
print(len(words), "words without stopwords and punctuations:", words)

from nltk import word_tokenize, pos_tag
text = "My favourite book is Ana Karenina"
tokens = word_tokenize(text)


y=pos_tag(tokens)
print(pos_tag(tokens))

from nltk import pos_tag, word_tokenize, RegexpParser
sample_text = "My favourite book is Ana Karenina"

# Find all parts of speech in above sentence
tagged = pos_tag(word_tokenize(sample_text))
  
#Extract all parts of speech from any text
chunker = RegexpParser("""
                       NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
                       P: {<IN>}               #To extract Prepositions
                       V: {<V.*>}              #To extract Verbs
                       PP: {<p> <NP>}          #To extract Prepositional Phrases
                       VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                       """)
 
# Print all parts of speech in above sentence
output = chunker.parse(tagged)
print("After Extracting\n", output)

output.draw()