import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
grammar = ('''NP: {<DT>?<JJ>*<NN>} # NP''')
chunkParser = nltk.RegexpParser(grammar)

paths = "interstellar_poem.txt"
with open(paths, "r") as f:
    txt = f.read()
f.close()

data = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(txt)]
delete = []

for sentence in data:
    delete = []
    for i in range(len(sentence)):
        sentence[i] = sentence[i].lower()
        sentence[i] = re.sub(r'\W', ' ', sentence[i])
        sentence[i] = re.sub(r'\s+', ' ', sentence[i])
        if sentence[i] == " ":
            delete.append(i)
    for k in reversed(delete):
        del sentence[k]
print(data)

wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"

for sentence in data:
    for w in sentence:
        if(w != wordnet_lemmatizer.lemmatize(w)):
            print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
        if(w != porter_stemmer.stem(w)):
            print("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))

tagged_data = []
for sentence in data:
    tagged = nltk.pos_tag(sentence)
    print(tagged)
    tagged_data.append(tagged)
for i in tagged_data:
    tree = chunkParser.parse(i)
    tree.draw()
for i in range(len(data)):
    print([w for w in data[i] if not w in stop_words])