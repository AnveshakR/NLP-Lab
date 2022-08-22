import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
string.punctuation = string.punctuation + '“' + '”' +'-' + '’' + '‘' + '—'
string.punctuation = string.punctuation.replace('.', '')
file = open('odessey.txt', encoding = 'utf8').read()
from nltk.util import ngrams
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#preprocess data

file_nl_removed = ""
for line in file:
    line_nl_removed = line.replace("\n", " ") #removes newlines
    file_nl_removed += line_nl_removed

file_p = "".join([char for char in file_nl_removed if char not in
string.punctuation]) #removes all special characters

sents = nltk.sent_tokenize(file_p)
print("The number of sentences is", len(sents)) #prints the number of sentences

words = nltk.word_tokenize(file_p)
print("The number of tokens is", len(words)) #prints the number of tokens

average_tokens = round(len(words)/len(sents))
print("The average number of tokens per sentence is", average_tokens) #prints the average number of tokens per sentence
unique_tokens = set(words)
print("The number of unique tokens are", len(unique_tokens)) #prints the number of unique tokens

unigram=[]
bigram=[]
trigram=[]
fourgram=[]
tokenized_text = []
for sentence in sents:
    sentence = sentence.lower()
    sequence = word_tokenize(sentence)
    for word in sequence:
        if (word =='.'):
            sequence.remove(word)
        else:
            unigram.append(word)
        tokenized_text.append(sequence)
    bigram.extend(list(ngrams(sequence, 2))) #unigram, bigram, trigram, and fourgram models are created
    trigram.extend(list(ngrams(sequence, 3)))
    fourgram.extend(list(ngrams(sequence, 4)))

def removal(x): #removes ngrams containing only stopwords
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in stop_words:
                count = count or 0
            else:
                count = count or 1
        if (count==1):
            y.append(pair)
    return(y)

bigram = removal(bigram)
trigram = removal(trigram)
fourgram = removal(fourgram)

freq_bi = nltk.FreqDist(bigram)
freq_tri = nltk.FreqDist(trigram)
freq_four = nltk.FreqDist(fourgram)

print("Most common n-grams without stopword removal and without add-1 smoothing: \n")
print ("Most common bigrams: ", freq_bi.most_common(5)) #prints most common n-grams without add-1 smoothing and without stopword removal.
print ("\nMost common trigrams: ", freq_tri.most_common(5))
print ("\nMost common fourgrams: ", freq_four.most_common(5))

#prints top 10 unigrams, bigrams after removing stopwords

print("Most common n-grams with stopword removal and without add-1 smoothing: \n")
unigram_sw_removed = [p for p in unigram if p not in stop_words]
fdist = nltk.FreqDist(unigram_sw_removed)
print("Most common unigrams: ", fdist.most_common(10))
bigram_sw_removed = []
bigram_sw_removed.extend(list(ngrams(unigram_sw_removed, 2)))
fdist = nltk.FreqDist(bigram_sw_removed)
print("\nMost common bigrams: ", fdist.most_common(10))

#Add-1 smoothing is performed here:

ngrams_all = {1:[], 2:[], 3:[], 4:[]}
for i in range(4):
    for each in tokenized_text:
        for j in ngrams(each, i+1):
            ngrams_all[i+1].append(j);

ngrams_voc = {1:set([]), 2:set([]), 3:set([]), 4:set([])}

for i in range(4):
    for gram in ngrams_all[i+1]:
        if gram not in ngrams_voc[i+1]:
            ngrams_voc[i+1].add(gram)

total_ngrams = {1:-1, 2:-1, 3:-1, 4:-1}
total_voc = {1:-1, 2:-1, 3:-1, 4:-1}
for i in range(4):
    total_ngrams[i+1] = len(ngrams_all[i+1])
    total_voc[i+1] = len(ngrams_voc[i+1])

ngrams_prob = {1:[], 2:[], 3:[], 4:[]}
for i in range(4):
    for ngram in ngrams_voc[i+1]:
        tlist = [ngram]
        tlist.append(ngrams_all[i+1].count(ngram))
        ngrams_prob[i+1].append(tlist)

for i in range(4):
    for ngram in ngrams_prob[i+1]:
        ngram[-1] = (ngram[-1]+1)/(total_ngrams[i+1] + total_voc[i+1]) #add-1 smoothing

#Prints top 10 unigrams, bigrams, trigrams, fourgrams after smoothing

print("Most common n-grams without stopword removal and with add-1 smoothing: \n")
for i in range(4):
    ngrams_prob[i+1] = sorted(ngrams_prob[i+1], key = lambda x:x[1], reverse = True)

print ("Most common unigrams: ", str(ngrams_prob[1][:10]))
print ("\nMost common bigrams: ", str(ngrams_prob[2][:10]))
print ("\nMost common trigrams: ", str(ngrams_prob[3][:10]))
print ("\nMost common fourgrams: ", str(ngrams_prob[4][:10]))

str1 = 'into that good night'
str2 = 'do not go gentle'

#smoothed models without stopwords removed are used

token_1 = word_tokenize(str1)
token_2 = word_tokenize(str2)

ngram_1 = {1:[], 2:[], 3:[]} #to store n-grams formed from the strings
ngram_2 = {1:[], 2:[], 3:[]}
for i in range(3):
    ngram_1[i+1] = list(ngrams(token_1, i+1))[-1]
    ngram_2[i+1] = list(ngrams(token_2, i+1))[-1]

print("String 1: ", ngram_1,"\nString 2: ",ngram_2)

for i in range(4):
    ngrams_prob[i+1] = sorted(ngrams_prob[i+1], key = lambda x:x[1], reverse = True)

pred_1 = {1:[], 2:[], 3:[]}

for i in range(3):
    count = 0
    for each in ngrams_prob[i+2]:
        if each[0][:-1] == ngram_1[i+1]: #to find predictions based on highest probability of n-grams
            count +=1
            pred_1[i+1].append(each[0][-1])
            if count ==5:
                break
    if count<5:
        while(count!=5):
            pred_1[i+1].append("NOT FOUND") #if no word prediction is found, replace with NOT FOUND
            count +=1

for i in range(4):
    ngrams_prob[i+1] = sorted(ngrams_prob[i+1], key = lambda x:x[1], reverse = True)

pred_2 = {1:[], 2:[], 3:[]}

for i in range(3):
    count = 0
    for each in ngrams_prob[i+2]:
        if each[0][:-1] == ngram_2[i+1]:
            count +=1
            pred_2[i+1].append(each[0][-1])
            if count ==5:
                break
    if count<5:
        while(count!=5):
            pred_2[i+1].append("\0")
            count +=1

print("\nNext word predictions for the strings using the probability models of bigrams,trigrams, and fourgrams\n")
print("String 1 - {}-\n".format(str1))
print("Bigram model predictions: {}\nTrigram model predictions: {}\nFourgram model predictions: {}\n" .format(pred_1[1], pred_1[2], pred_1[3]))
print("String 2 - {}-\n".format(str2))
print("Bigram model predictions: {}\nTrigram model predictions: {}\nFourgram model predictions: {}" .format(pred_2[1], pred_2[2], pred_2[3]))