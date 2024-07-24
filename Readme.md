# Image Classification with CIFAR-10 Dataset

This project is a web application that uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. It's built with TensorFlow and Flask, providing both a backend API and a frontend interface for image classification.

## Features

- Train a CNN model on the CIFAR-10 dataset
- Classify uploaded images into one of 10 categories
- View model training history
- Evaluate model performance
- Save and load trained models

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone this repository:

2. Install the required packages:

## Usage

1. Start the Flask server:

2. Open a web browser and navigate to `http://localhost:5000`

3. Use the web interface to:
- Upload and classify images
- View model accuracy
- Save or load the model

## API Endpoints

- `POST /predict`: Classify an uploaded image
- `GET /evaluate`: Get the model's accuracy on the test set
- `GET /training_history`: Get the model's training history
- `POST /save_model`: Save the current model
- `POST /load_model`: Load a saved model

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with BatchNormalization and MaxPooling
- 1 Dense layer with Dropout
- Output layer for 10 classes

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b <branch_name>`
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request

## License

This project uses the following license: [MIT License](https://opensource.org/licenses/MIT).

## Contact

If you want to contact me, you can reach me at `<your_email@example.com>`.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
