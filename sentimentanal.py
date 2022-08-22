import tensorflow as tf
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
(train_data_raw, train_labels), (test_data_raw, test_labels) = tf.keras.datasets.imdb.load_data(index_from=3)
words2idx = tf.keras.datasets.imdb.get_word_index()
idx2words = {idx:word for word, idx in words2idx.items()}
imdb_reviews = []
for review, label in zip(train_data_raw, train_labels):
    try:
        tokens = [idx2words[x-3] for x in review[1:]]
        text = ' '.join(tokens)
        imdb_reviews.append([text, label])
    except: # There is a distorted observation. For that, we need to handle the error
        print('Small index number')
        pass

imdb_df = pd.DataFrame(imdb_reviews,columns=['Text', 'Label'])
print(imdb_df.info())
print(imdb_df.head(10))

sentences = ['Hello, world. I am terrible']
for sentence in sentences:
    print(sentence)
    ss = sia.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')

imdb_slice = imdb_df.sample(frac=1.0).reset_index(drop=True)
imdb_slice['Prediction'] = imdb_slice['Text'].apply(lambda x: 1 if
sia.polarity_scores(x)['compound'] >= 0 else -1)
imdb_slice['Label'] = imdb_slice['Label'].apply(lambda x: -1 if x == 0 else 1)
imdb_slice['Accuracy'] = imdb_slice.apply(lambda x: 1 if x[1] == x[2] else 0, axis=1)

def conf_matrix(x):
    if x[1] == 1 and x[2] == 1:
     return 'TP'
    elif x[1] == 1 and x[2] == -1:
        return 'FN'
    elif x[1] == -1 and x[2] == 1:
        return 'FP'
    elif x[1] == -1 and x[2] == -1:
        return 'TN'
    else:
        return 0
        
imdb_slice['Conf_Matrix'] = imdb_slice.apply(lambda x: conf_matrix(x), axis=1)
imdb_slice.tail(10)
conf_vals = imdb_slice.Conf_Matrix.value_counts().to_dict()
print(conf_vals)
accuracy = (conf_vals['TP'] + conf_vals['TN']) / (conf_vals['TP'] + conf_vals['TN'] +
conf_vals['FP'] + conf_vals['FN'])
precision = conf_vals['TP'] / (conf_vals['TP'] + conf_vals['FP'])
recall = conf_vals['TP'] / (conf_vals['TP'] + conf_vals['FN'])
f1_score = 2*precision*recall / (precision + recall)
print('Accuracy: ', round(100 * accuracy, 2),'%',
'\nPrecision: ', round(100 * precision, 2),'%',
'\nRecall: ', round(100 * recall, 2),'%',
'\nF1 Score: ', round(100 * f1_score, 2),'%')