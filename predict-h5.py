# load model
from keras.models import load_model
from numpy import loadtxt


model = load_model('models/model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))