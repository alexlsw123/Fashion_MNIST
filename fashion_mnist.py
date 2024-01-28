import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Setting the seed
tf.keras.utils.set_random_seed(38614)

tf.config.experimental.enable_op_determinism()

# Loading and processing the data
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = 
mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images/255, test_images/255

# Model creation
model = tf.keras.Sequential([
 tf.keras.layers.Conv2D(64, (3,3), kernel_initializer= 'he_normal', 
padding='same', activation='relu', input_shape=(28,28,1)),
 tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Dropout(0.5),
 tf.keras.layers.Conv2D(128, (3,3), kernel_initializer = 'he_normal',  
padding = 'same', activation = 'relu', input_shape=(28,28,1)),
 tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Dropout(0.5),
 tf.keras.layers.Conv2D(256, (3,3), kernel_initializer = 'he_normal',  
padding = 'same', activation = 'relu', input_shape=(28,28,1)),
 tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Dropout(0.5),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(2048, activation = 'relu'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.Dropout(0.5),
 tf.keras.layers.Dense(1028, activation = 'relu'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.Dense(10, activation = 'softmax')
])

# Optimizer with Adam
model.compile(optimizer = 'adam', loss = 
'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Learn rate modification
# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 
3, min_lr = 0.00001)

history = model.fit(train_images, train_labels, batch_size = 32, epochs = 
50, verbose = 1, validation_data = (test_images, test_labels), callbacks = 
[reduce_lr])

score = model.evaluate(test_images, test_labels, verbose=0)
print('\n', 'Test accuracy:', score[1])

# Plotting Model Accuracy plot for train and test image
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'lower right')

plt.savefig('Model_Accuracy.png')
