import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# some utils images = https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip 
base_dir = "your_base_dir"

## Import your own images dataset
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


train_cats_dir = os.path.join(base_dir, "cats")
train_dog_dir = os.path.join(base_dir, "dogs")

validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dog_dir = os.path.join(validation_dir, "dogs")

train_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_datagen = ImageDataGenerator(rescale = 1/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)