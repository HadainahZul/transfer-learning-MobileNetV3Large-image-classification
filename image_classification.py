# %%
#1. Import packages
from tensorflow.keras import layers, optimizers, losses, metrics, callbacks, applications
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# %%
#2. Load dataset
PATH = os.path.join(os.getcwd(), 'dataset')

#(A) Define the path to the train and test data folder
#train_path = os.path.join(PATH, 'Positive')
#test_path = os.path.join(PATH, 'Negative')

#(B) define the batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# %%
#(C) Load data into tensorflow dataset using the specific method
train_dataset = keras.utils.image_dataset_from_directory(PATH, validation_split=0.3, subset='training', seed = 42, shuffle = True, batch_size = BATCH_SIZE, image_size = IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(PATH, validation_split=0.3, subset='validation', seed =42, shuffle = True, batch_size = BATCH_SIZE, image_size = IMG_SIZE)
class_names = train_dataset.class_names

# %%
#3. Convert BatchDataset to PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size = AUTOTUNE)

# %%
#4. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#Apply the data augmentation to test it out
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis = 0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.axis('off')
        
# %%
#5. Apply transfer learning
#(A) Import MobileNetV3Large

IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = keras.applications.MobileNetV3Large(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet', pooling = 'avg')
feature_extractor.summary()
keras.utils.plot_model(feature_extractor, show_shapes = True)

# %%
#6. Define the classification layer
l2 = keras.regularizers.L2()
output_layer = layers.Dense(len(class_names), activation='softmax', kernel_regularizer = l2)

# %%
#7. Use fuctional API to create the entire model pipeline
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = feature_extractor(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# %%
#8. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# %%
#Find loss and accuracy
loss0, acc0 = model.evaluate(pf_test)
print("Loss = ", loss0)
print("Accuracy = ", acc0)

# %%
#10. Create tensorboard callback
import datetime
log_path = os.path.join('log_dir', 'ass_3', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir = log_path)

# %%
#11. Model training
EPOCHS = 5
history = model.fit(pf_train, validation_data = pf_test, epochs = EPOCHS, callbacks = [tb])

# %%
#12. Apply model fine tuning strategy
feature_extractor.trainable = True

for layer in feature_extractor.layers[:132]:
    layer.trainable = False

model.summary()

# %%
#13. Model compile
optimizer = optimizers.RMSprop(learning_rate = 0.00001)
model.compile(optimizer=optimizer, loss = loss, metrics = ['accuracy'])

# %%
#14. Continue the model training
fine_tune_epoch = 5
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(pf_train, validation_data = pf_test, epochs = total_epoch, initial_epoch = history.epoch[-1], callbacks = [tb])

# %%
#15. Evaluate the model after training
test_loss, test_acc = model.evaluate(pf_test)

print("Loss = ", test_dataset)
print("Accuracy = ", test_acc)

# %%
#16. Model Deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

plt.figure(figsize=(20,20))
for i in range(len(image_batch)):
    plt.subplot(8,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"Label: {label_batch[i]}, Prediction: {y_pred[i]}")
    plt.axis('off')
plt.show()

# %%
#Compare label and predictions
label_vs_prediction = np.transpose(np.vstack((label_batch, y_pred)))

print(label_vs_prediction)

# %% Save model
# To save trained model
model.save('saved_model', 'model.h5')
model.save('model.h5')
# %%
