import os 
from nltk import word_tokenize, WordNetLemmatizer, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
import random 
from collections import Counter

EMAIL_PATH = '/enron2/' 
stoplist = stopwords.words('english')

#--------------------
#function definitions
#--------------------
 
def initalize_data(path): 
	dataList = [] 
	try:
		fileList = os.listdir(path)
		for aFile in fileList: 
			f = open(path + aFile, 'r')
			dataList.append(f.read())
		f.close()
	except: 
		print('There was an error loading the data')
	return dataList
	
def preprocess(sentence): 
	lemmatizer = WordNetLemmatizer()
	tokenized = word_tokenize(sentence) 
	return [lemmatizer.lemmatize(word.lower()) for word in tokenized] 
	
def extract_features(text): 
	return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist} 

def train(features, trainingProportion, classifierType):
	trainCount = int(len(features) * trainingProportion)
	trainEmails, testEmails = features[:trainCount], features[trainCount:]
	print('Training set size = ' + str(len(trainEmails)))
	print('Test set size = ' + str(len(testEmails))) 
	
	if classifierType == 'NaiveBayes': 
		classifier = NaiveBayesClassifier.train(trainEmails) 
	else :
		classifier = WekaClassifier.train(trainEmails) 
	return trainEmails, testEmails, classifier



#-------------------
#MAIN
#-------------------
#Initalize data into lists, then create tuples to keep track of which emails are spam vs ham 
spamList = initalize_data(EMAIL_PATH + 'spam/')
hamList = initalize_data(EMAIL_PATH + 'ham/')
spamEmails = [(email, 'spam') for email in spamList]
hamEmails = [(email, 'ham') for email in hamList]

allEmails = spamEmails + hamEmails #combine spam and ham and shuffle 
random.shuffle(allEmails) 

#Extracting and preprocessing data 
features = [(extract_features(email), label) for (email, label) in allEmails]

#training classifier 
trainEmails, testEmails, classifier = train(features, 0.8, 'NaiveBayes') 
#trainEmails, testEmails, classifier = train(features, 0.8, 'Weka') 

#evaluate
print('Accuracy of training emails = ' + str(classify.accuracy(classifier, trainEmails)))
print('Accuracy of test emails = ' + str(classify.accuracy(classifier, testEmails))) 
classifier.show_most_informative_features(20) 





