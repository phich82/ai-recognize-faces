# load pima indians dataset
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from numpy import loadtxt

""" Protocol Buffer Format (.pb)
    It is considered faster to save and load a protocol buffer format, but doing so will produce multiple files.

    model/
    |-- assets/
    |-- keras_metadata.pb
    |-- saved_model.pb
    `-- variables/
        |-- variables.data-00000-of-00001
        `-- variables.index
"""

dataset = loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("models/pb")
print("Saved model to disk")

# load model
model = load_model('models/pb')
# print summary
model.summary()