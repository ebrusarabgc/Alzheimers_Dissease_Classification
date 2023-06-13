import tensorflow as tf
from data_utils import load_data
from model_utils import create_model, compile_model, train_model, evaluate_model

# Constants
data_dir = "Dataset"
val_split = 0.2
image_size = (128, 128)
batch_size = 32
color_mode = "grayscale"
seed = 0
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
learning_rate = 0.001
epochs = 50

# Load and preprocess the data
train_data, valid_data, class_names = load_data(data_dir, val_split, image_size, batch_size, color_mode, seed)

# Create the model
model = create_model(image_size + (1,), base_model, len(class_names))

# Compile the model
compile_model(model, learning_rate)

# Train the model
train_model(model, train_data, valid_data, epochs)

# Evaluate the model
evaluate_model(model, valid_data)

# Print the class names
print(class_names)

# Print the model summary
model.summary()
