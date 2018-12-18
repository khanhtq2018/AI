# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

# Part 2 - Fitting the CNN to the images

# Check whether trained model
import os  
if os.path.isfile('model.h5'):
    classifier.load('model.h5')
    print('The model is trained in last time')
else:
    print('Starting build the model...')
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                     shear_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('training_set',
                                                   target_size = (224, 224),
                                                   batch_size = 32,
                                                   class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('test_set',
                                              target_size = (224, 224),
                                              batch_size = 32,
                                              class_mode = 'binary')
    classifier.fit_generator(training_set,
                           steps_per_epoch = 1200,
                           epochs = 25,
                           validation_data = test_set,
                           validation_steps = 300)
    classifier.save('model.h5')
    print('The model is trained and saved successfully')

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('prediction_set/NoPerson/1501.jpeg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Person'
else:
    prediction = 'NoPerson'

print(prediction)