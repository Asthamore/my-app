#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import cv2
from sklearn.utils.class_weight import compute_class_weight


# In[5]:


train_dir = "C:\\Users\\DELL\\Downloads\\archive\\Training"
test_dir = "C:\\Users\\DELL\\Downloads\\archive\\Testing"

# Augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)
class_weights = dict(enumerate(class_weights))


# In[7]:


# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
# Unfreeze the last 50 layers of ResNet50
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Learning rate scheduler
def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 5:
        lr = 1e-5
    elif epoch > 10:
        lr = 1e-6
    print("Learning rate: ", lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train and test generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the model
model.save("brain_tumor_model.h5")


# In[ ]:





# In[ ]:





# In[10]:


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import os

# Load your model from epoch 10
model = load_model("brain_tumor_model.h5")

# Unfreeze more layers for fine-tuning (e.g., last 70 layers)
for layer in model.layers[-70:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler: tiny bump down mid-fine-tuning
def lr_schedule(epoch):
    return 1e-5 if epoch < 2 else 1e-6

# Callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Data generators
train_dir = "C:\\Users\\DELL\\Downloads\\archive\\Training"
test_dir = "C:\\Users\\DELL\\Downloads\\archive\\Testing"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                                  batch_size=32, class_mode='categorical')

# Continue training (epochs 11–15)
model.fit(train_generator,
          validation_data=test_generator,
          epochs=15,
          initial_epoch=10,
          callbacks=[lr_scheduler, early_stop],
          verbose=1)

# Save updated model
model.save("brain_tumor_model_finetuned_epoch15.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


pip install streamlit


# In[5]:


import os
os.getcwd()


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model (after training or fine-tuning)
model = load_model("brain_tumor_model_finetuned_epoch15.h5")

# Define the class labels
class_labels = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

# Function to preprocess the input image (rescale, resize, etc.)
def preprocess_image(img_path):
    # Load the image with the required target size for the model
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Normalize the image (same as you did during training)
    img_array = img_array / 255.0
    
    # Add an extra batch dimension (this is required for the model)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict the class of the input image
def predict_tumor_type(img_path):
    img_array = preprocess_image(img_path)
    
    # Make a prediction (get the probabilities for each class)
    prediction = model.predict(img_array)
    
    # Show raw probabilities (useful for debugging)
    print("Raw probabilities:", prediction)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    # Map it to the corresponding label
    predicted_class = class_labels[predicted_class_index]
    
    print(f"Predicted class: {predicted_class}")
    
    # Return the class and confidence level
    confidence = prediction[0][predicted_class_index] * 100
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class, confidence

# Testing the function: Ask user for an image file path
img_path = input("Enter the path to the image for prediction: ")

# Get the prediction result
predicted_class, confidence = predict_tumor_type(img_path)

# Optionally, print the top-3 most likely classes with their probabilities
def top_3_predictions(img_path):
    img_array = preprocess_image(img_path)
    
    # Get probabilities for each class
    probs = model.predict(img_array)[0]
    
    # Get the top 3 predicted classes
    top_3 = np.argsort(probs)[-3:][::-1]
    
    print("Top 3 predictions:")
    for i in top_3:
        print(f"{class_labels[i]}: {probs[i]*100:.2f}%")
    
top_3_predictions(img_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




