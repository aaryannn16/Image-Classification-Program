import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Loading pre-trained ResNet50V2 model without the top classification layer
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling = 'None')

# Adding new layers on top of the pre-trained base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = BatchNormalization()(x)  # Normalize activations
x = Dropout(0.5)(x)  # Dropout to prevent overfitting

x = Dense(512, activation='relu')(x)  # Another fully connected layer
x = BatchNormalization()(x)  # Normalize activations
x = Dropout(0.5)(x)  # Another dropout to improve regularization

# Output layer for 3 classes (for face classification)
predictions = Dense(3, activation='softmax')(x)  # Softmax for multi-class classification

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                   shear_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Data preparation for validation data (usually without augmentation, only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    'Training_Dataset',  # Add the path of your Training dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    'Validation_Dataset',  # Add the path of your validation dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Training the model with validation dataset
model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Saving the model
model.save('image_classification_model.h5')
