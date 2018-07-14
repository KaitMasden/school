import os


#functions
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



#MAIN
tweets = initalize_data('C:\\Users\\migue\\Documents\\Kait\\Text Clustering Mini\\Health-Tweets\\cbchealth.txt')