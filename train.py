# Ref: https://www.section.io/engineering-education/image-classifier-keras/

import os
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow
import matplotlib.pyplot as plt

# Create batches of images data
dgen_train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # using 20% of training data for validation
    zoom_range=0.2,
    horizontal_flip=True
)
dgen_validation = ImageDataGenerator(rescale=1./255)
dgen_test = ImageDataGenerator(rescale=1./255)


# HyperParameters
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE = 'binary' # for two classes
# CLASS_MODE = 'categorical' # for over 2 classes

train_dir = os.path.join('./data/training', 'train')
val_dir = os.path.join('./data/training', 'val')
test_dir = os.path.join('./data/training', 'test')

# # Directory with train taylor images
# train_taylor_dir = os.path.join(train_dir, 'taylor')
# train_celedion_dir = os.path.join(train_dir, 'celedion')
# # Directory with test taylor image
# test_taylor_dir = os.path.join(test_dir, 'taylor')
# # Directory with test celedion image
# test_celedion_dir = os.path.join(test_dir, 'celedion')

# Connecting the ImageDataGenerator objects to dataset
train_generator = dgen_train.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    subset='training',
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE
)
validation_generator = dgen_train.flow_from_directory(
    val_dir,
    target_size=TARGET_SIZE,
    subset='validation',
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE
)
test_generator = dgen_test.flow_from_directory(
    test_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE
)

# Get the class indices
print('class_indices => ', train_generator.class_indices)

# Building CNN Model (image input sizse = 200 x 200)
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# Display the model's architecture
model.summary()

# Compile the Model
# binary_crossentropy loss: for two classes
model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# categorical_crossentropy loss: for more than two classes
# model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
# history = model.fit(
#     train_generator,
#     epochs=30,
#     # validation_data=validation_generator,
#     validation_data=test_generator,
#     callbacks=[
#         # Stopping our training if val_accuracy doesn't improve after 20 epochs
#         # tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20), # accuracy, loss
#         tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20), # accuracy, loss
#         # Saving the best weights of our model in the model directory

#         # We don't want to save just the weight, but also the model architecture
#         tensorflow.keras.callbacks.ModelCheckpoint(
#             # 'models/model_{accuracy:.3f}.h5', # accuracy, loss
#             'models/model_{val_accuracy:.3f}.h5', # accuracy, loss
#             save_best_only=True,
#             save_weights_only=False,
#             # monitor='accuracy' # accuracy, loss
#             monitor='val_accuracy' # accuracy, loss
#         )
#     ]
# )

#############################################################
model.fit(train_generator, validation_data=test_generator, epochs=30, batch_size=2, verbose=0, use_multiprocessing=False)
# evaluate the model
scores = model.evaluate(train_generator, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# model.save("models/model_last.h5")
model.save("models/model_last.keras")
print("Saved model to disk")
#############################################################

# Performance Evaluation
# print(history.history.keys())

# Plot graph between training and validation loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['Training', 'Validation'])
# plt.title('Training and Validation Losses')
# plt.xlabel('epoch')
