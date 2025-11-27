import tensorflow as tf 
from tensorflow.keras import models , layers
import matplotlib.pyplot as plt
import os
import numpy as np  
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import keras

IMAGE_SIZE=224
batch_size = 32
channel = 3
epochs=10


# Define parameters
IMAGE_SIZE = 224  # Set your desired image size
BATCH_SIZE = 32   # Set your batch size

# Path to dataset (adjust the path accordingly)
dataset_path = os.path.join(os.getcwd(), 'Mix Both Datasets')

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

train_ds = 0.8
len(dataset)*train_ds

test_ds=dataset.skip(24)

val_size = test_ds.take(6)

# Check dataset class names
class_names = dataset.class_names
print("Class names:", class_names)

class_name=dataset.class_names
class_name

for image_batch , label_batch in dataset.take(1):
    print(image_batch.numpy().shape)

plt.figure(figsize = (10,10))
for image_batch , label_batch in dataset.take(1):
    for i in range(12):
        ax=plt.subplot(3 , 4 , i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_name[label_batch[i]])
        plt.axis("off")    
plt.show()


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

resizing_rescaling = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE , IMAGE_SIZE),
    layers.Rescaling(1.0/255)

])

data_augmentaion = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)

])

N_CLASSES = 2     # Number of classes

# Define input shape for the first layer
input_shape = (IMAGE_SIZE, IMAGE_SIZE, channel)  # Removed batch_size



# ResNet50 Model (replaces EfficientNetB0 to avoid the weight shape bug)

base_model = ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                      include_top=False,
                      weights='imagenet')
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(N_CLASSES, activation='softmax')
])


# Model summary to check the architecture
model.summary()

model.compile(
    optimizer = "adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ["accuracy"])

history = model.fit(
    train_ds,
    epochs=epochs,
    batch_size = batch_size,
    verbose = 1,
    validation_data = val_ds)

score = model.evaluate(test_ds)

# Add confusion matrix code below

# Generate predictions and true labels
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

score=model.predict(train_ds)
predicted_labels = np.argmax(score, axis=1)
predicted_labels



# Extracting metrics from history
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Determining the number of epochs
epochs = range(len(acc))

# Plotting training and validation accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

for image_batch , label_batch in test_ds.take(1):
    first_image =(image_batch[0].numpy().astype ("uint8"))
    first_label= label_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual_label:" , class_name[first_label])

    batch_prediction=model.predict(image_batch)
    print("predicted_label:" , class_name[np.argmax(batch_prediction[0])])



def predict(model , img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array , 0)

    prediction=model.predict(img_array)

    prediction_class=class_name[np.argmax(batch_prediction[0])]
    confidence=round(100*(np.max(prediction[0])) , 2)
    return prediction_class , confidence

print(f"Label index: {label_batch[0]}, Actual class: {class_name[label_batch[0]]}")


plt.figure(figsize=(13, 13))

# Taking one batch from the test dataset
for images, labels in test_ds.take(1):
    # Iterating through the first 9 images in the batch
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)  # Create a 3x3 grid of subplots
        plt.imshow(images[i].numpy().astype("uint8"))  # Display the image

        # Predict the class and confidence
        predicted_class, confidence = predict(model, images[i].numpy())

        # Get the actual class name from labels
        actual_class = class_name[labels[i]]

        # Set the title of each subplot
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence:.2f}%")
        plt.axis("off")  # Turn off the axis

plt.tight_layout()  # Adjust layout for better display
plt.show()

model_version = 5
model.save(f"model_{model_version}.keras")  # Add .keras extension
m = keras.models.load_model("model_5.keras")
m.save("model_5.h5")  # legacy format that TF 2.15 can load

for i in range(5):
    actual_class = class_name[label_batch[i].numpy()]
    predicted_class, confidence = predict(model, image_batch[i].numpy())
    print(f"Actual: {actual_class}, Predicted: {predicted_class}, Confidence: {confidence:.2f}%")