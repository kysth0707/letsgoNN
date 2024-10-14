def singleUnique(data):
	"""
	change str to int
	"""
	changeDict = {x : i for i, x in enumerate(set(data))}
	return (
		changeDict,
		[changeDict[d] for d in data]
	)

def multipleUnique(data):
	"""
	change str to int
	"""
	uniqueCount = len(data[0])
	tmpData = [[] for _ in range(uniqueCount)]

	for d in data:
		for i in range(uniqueCount):
			tmpData[i].append(d[i])
	
	changeDicts = [{x : i for i, x in enumerate(set(dataSplit))} for dataSplit in tmpData]
	print(changeDicts)

	return (
		changeDicts,
		[
			[changeDicts[i][d[i]] for i in range(uniqueCount)] for d in data
   		]
	)

rawTexts = []
with open('mushrooms.csv', 'r', encoding='utf8') as f:
	rawTexts = f.readlines()

dataX = []
dataY = []

tmp = rawTexts[1].rstrip().split(',')

for rawTxt in rawTexts[1:]:
	k = rawTxt.rstrip().split(',')
	dataX.append(k[1:])
	dataY.append(k[0])


dataXChangeDict, dataX = multipleUnique(dataX)
dataYChangeDict, dataY = singleUnique(dataY)