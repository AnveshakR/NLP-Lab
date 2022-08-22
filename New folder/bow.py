from nbformat import write
import nltk
import re
import numpy as np
import heapq
import csv

paths = "mary_rhyme.txt"

with open(paths, "r") as f:
    txt = f.readline()

f.close()

data = nltk.sent_tokenize(txt)

for i in range(len(data)):
    data[i] = data[i].lower()
    data[i] = re.sub(r'\W', ' ', data[i])
    data[i] = re.sub(r'\s+', ' ', data[i])

wordcount = {}
for i in data:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in wordcount.keys():
            wordcount[word] = 1
        else:
            wordcount[word] +=1

print(wordcount)
print(wordcount.__len__())

print("\n\n")

freq_words = heapq.nlargest(100, wordcount, key=wordcount.get)
X = []

for data in data:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
X = np.asarray(X)

print(X)

with open('wordcount.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for key,value in wordcount.items():
        writer.writerow([key,value])

with open("wordvector.csv",'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(X)