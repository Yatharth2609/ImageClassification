import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Global variables
model = None
history = None

def create_and_train_model():
    global model, history
    
    # Load and preprocess the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Learning rate scheduler
    def lr_schedule(epoch):
        return 0.001 * (0.1 ** int(epoch / 10))

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # Train the model
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[lr_scheduler]
    )

    return model, history

def load_or_create_model():
    global model, history
    model_path = 'saved_model.keras'
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training new model...")
        model, history = create_and_train_model()
        model.save(model_path)

# Load or create the model when the application starts
load_or_create_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence = float(tf.nn.softmax(prediction)[0][predicted_class])
    return jsonify({
        'class': class_names[predicted_class],
        'confidence': confidence
    })

@app.route('/evaluate')
def evaluate():
    _, (test_images, test_labels) = cifar10.load_data()
    test_images = test_images / 255.0
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    return jsonify({'accuracy': float(test_acc)})

@app.route('/training_history')
def training_history():
    if history is None:
        return jsonify({'error': 'No training history available'}), 404
    
    return jsonify({
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })

@app.route('/save_model', methods=['POST'])
def save_model():
    model_path = 'saved_model.keras' 
    model.save(model_path)
    return jsonify({'message': f'Model saved to {model_path}'})

@app.route('/load_model', methods=['POST'])
def load_model():
    global model
    model_path = 'saved_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return jsonify({'message': f'Model loaded from {model_path}'})
    else:
        return jsonify({'error': 'No saved model found'}), 404

if __name__ == '__main__':
    app.run(debug=True)