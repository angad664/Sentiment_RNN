import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#imdb dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words = 10000, valid_portion=0.1)

trainX, trainY = train
testX, testY = test

# data processing
#squence padding - transform list into 2d numpy array
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
#to_categorical - converts label to binary vectors with 2 classes with  0= -ve and 1 is +ve
trainY = to_categorical(trainY, nb_classes = 2)
testY = to_categorical(testY, nb_classes = 2)

#network building
net = tflearn.input_data([None,100]) # maxlen is 100 that means our batch size is 100
net = tflearn.embedding(net, input_dim=10000, output_dim= 128) # 10000 bc thats the most we want and 128 bc
# thats the result of our embedding
#lstm= long short term memory and dropout = to prevent from overfitting (to drop some of nodes)
net = tfkearn.lstm(net,128, dropout=0.8)
#fully_connected= means that they are connected to every neuron in the layer and softmax = probability range
#between 0 to 1
net = tflearn.fully_connected(net, 2, activation='softmax')
# learning_rate= how fast you want our network to train, adam perform gradient descent ,
#loss = categorical_crossentropy it helps to find difference between expected and predicted
net = tflearn.regression(net,optimizer ='adam', learning_rate =0.001, loss ='categorical_crossentropy')

#training
#DNN = deep neural network
model = tflearn.DNN(net, tensorboard_verbose = 0)
#show_metric = true bc  do the log of accuracy during training
model.fit(trainX, trainY, validation_set=(testX,testY),show_metric = True, batch_size = 32)
