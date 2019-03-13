from operator import itemgetter
from keras.models import load_model
from inputHandler import *
from model import SiameseBiLSTM
from config import siamese_config
import pandas as pd
from sklearn.metrics import accuracy_score

best_model_path = "checkpoints/1551846625/" + str("lstm_50_50_0.17_0.25.h5")
model = load_model(best_model_path)

df = pd.read_csv('c_skeletons_test.csv')

sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
is_similar = list(df['is_similar'])
del df


sentences1 = [str(sent) for sent in sentences1]
sentences2 = [str(sent) for sent in sentences2]

tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}

#test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?'), ('What can make Physics easy to learn?','What does it mean that every time I look at the clock the numbers are the same?')]

test_sentence_pairs = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs, siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]

pred_labels = []

for prediction, actual in zip(preds, is_similar):
	if(prediction > 0.50):
		pred_labels.append(1)
	else:
		pred_labels.append(0)

accuracy = accuracy_score(pred_labels, is_similar)
print ("Accuracy = " + str(accuracy))

#results.sort(key=itemgetter(2), reverse=True)
#print (results)