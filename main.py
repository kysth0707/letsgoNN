import math

# Model Sturcture 구현
# Activation Function 구현
# Weight 관리 ( bias 는 일단 빼고 )


# Variable 약속
# Class : 대소대소 ex) ClassLikeBMW
# Variable, Function : 소대소 ex) somethingGood



# Activation Function
class ActivationFunctions:
	def relu(self, x : float):
		"""
		x ( x > 0 )\n
		0 ( else )
		"""
		return x if x > 0 else 0
	
	def sigmoid(self, x : float):
		"""
		-inf ~ +inf => 0 ~ 1
		"""
		return 1 / (1 + math.exp(-x))
	
	def tanh(self, x : float):
		"""
		-inf ~ +inf => -1 ~ 1
		"""
		eP = math.exp(x) # exp Positive
		eN = math.exp(-x) # exp Negative
		return (eP - eN) / (eP + eN)

# Model Function
class ModelFunctions:
	def addDense(self, inputShape : int, activationFunction):
		return {
			'inputShape' : inputShape,
			'activationFunction' : activationFunction
		}

# Get Data And Preprocessing
def singleUnique(data):
	"""
	change str to int\n
	ex) q, k, q => {q : 0, k : 1}, [0, 1, 0]
	"""
	changeDict = {x : i for i, x in enumerate(set(data))}
	return (
		changeDict,
		[changeDict[d] for d in data]
	)

def multipleUnique(data):
	"""
	change str to int\n
	ex) [[q,k], [q,q]] => [{q : 0}, {q:0,k:1}], [[0, 1], [0, 0]]
	"""
	uniqueCount = len(data[0])
	tmpData = [[] for _ in range(uniqueCount)]

	for d in data:
		for i in range(uniqueCount):
			tmpData[i].append(d[i])
	
	changeDicts = [{x : i for i, x in enumerate(set(dataSplit))} for dataSplit in tmpData]

	return (
		changeDicts,
		[
			[changeDicts[i][d[i]] for i in range(uniqueCount)] for d in data
   		]
	)

# get Raw Text
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

# Preprocess It
dataXChangeDict, dataX = multipleUnique(dataX)
dataYChangeDict, dataY = singleUnique(dataY)

# Let's model~~~~~~
modelFunction = ModelFunctions()
activationFunction = ActivationFunctions()

ModelStructure = [
	modelFunction.addDense(len(dataX[0]), activationFunction.relu),
	modelFunction.addDense(16, activationFunction.relu),
	modelFunction.addDense(16, activationFunction.sigmoid)
]


# make Weight