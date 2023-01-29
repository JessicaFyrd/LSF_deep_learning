#cnn.py

#Import Librairies	=============================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
import keras
from keras import layers
from tensorflow.keras import regularizers


#Generate the datasets	=========================================================================================================================
#	Define the paths to the dataset.
# ~ training_path 	= '/Users/jess/Desktop/GE/5A_GE/S9/Vision/Projet/Projet_langue_signe/Langues_des_signes/Photos_train/'
training_path 	= '/Users/jess/Desktop/GE/5A_GE/S9/Vision/Projet/Projet_langue_signe/Langues_des_signes/Photos_train_data_aug/'
test_path 		= '/Users/jess/Desktop/GE/5A_GE/S9/Vision/Projet/Projet_langue_signe/Langues_des_signes/Photos_test/'

#	Create dataset
image_size = (28, 28)
batch_size = 32

training_set 		= tf.keras.utils.image_dataset_from_directory(
    directory		= training_path,
    labels			= "inferred",			#Labels are generated from the directory structure
    label_mode		= "int",  				#Labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss)
    class_names		= ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"],
    image_size		= image_size,
    validation_split= 0.15,					#15% data for validation set
    batch_size		= batch_size,
    color_mode		= 'grayscale',
    seed			= 123,
    subset			= "training"
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    directory		= training_path,
    labels			= "inferred",
    label_mode		= "int", 
    class_names		= ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"],
    image_size		= image_size,
    validation_split= 0.15,
    batch_size		= batch_size,
    color_mode		= 'grayscale',
    seed			= 123,
    subset			= "validation"
)

test_set = tf.keras.utils.image_dataset_from_directory(
    directory		= test_path,
    labels			= "inferred",
    label_mode		= "int", 
    image_size		= image_size,
    batch_size		= batch_size,
    color_mode		= 'grayscale'
)


#Build a model		=============================================================================================================================
model = keras.Sequential([
    # preprocessing
    layers.Rescaling(scale=1./255),
    
    # convolution 2D/Pooling
    layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', padding='same', input_shape=((image_size), 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu', padding='same'),
    
    layers.Flatten(),
    
    # network
    layers.Dense(84, activation='relu'),

    # output layer
    layers.Dense(11, activation='softmax')
])


#Callbacks 			=============================================================================================================================
epochs = 12
callbacks = [ # callbacks (save the model at each epoch)
    keras.callbacks.ModelCheckpoint("checkpoints/model_video_at_{epoch}.h5"),
]


#Set loss, metrics and otpimizer	=============================================================================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


#Fitting the model 	=============================================================================================================================
#	Fit it
print("\nFit model on training data")
history = model.fit(training_set, validation_data=validation_set, epochs=epochs, callbacks=callbacks, verbose=2)

#	Display the CNN
model.summary()

#	Plot it
plt.figure(figsize=(24, 8))
plt.title('Visualization of the model learning process', fontsize=14)
plt.plot(np.arange(1, epochs + 1), history.history['accuracy'], label='The fraction of correct answers on the training set') 
plt.plot(np.arange(1, epochs + 1), history.history['val_accuracy'], label='The fraction of correct answers of the validation set')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('The fraction of correct answers', fontsize=14)
plt.xticks(np.arange(1, epochs + 1), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1, epochs)
plt.grid()
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(24, 8))
plt.title('Visualization of the loss model', fontsize=14)
plt.plot(np.arange(1, epochs + 1), history.history['loss'], label='Loss on the training set') 
plt.plot(np.arange(1, epochs + 1), history.history['val_loss'], label='Loss of the validation set')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(np.arange(1, epochs + 1), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1, epochs)
plt.grid()
plt.legend(fontsize=14)
plt.show()


#Evaluate the model 	=========================================================================================================================
print("\nEvaluate on test data")
results = model.evaluate(test_set, verbose=0)
print("test loss, test acc:", results)


#Save the model 	=============================================================================================================================
# ~ model.save("cnn_model_video.h5")
model.save("cnn_data_aug_model_video.h5")

