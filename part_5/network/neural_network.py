import tensorflow as tf

path_to_train_dataset = "dataset/train"
path_to_test_dataset = "dataset/test"

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(path_to_train_dataset,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(28, 28),
                                                    color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(path_to_test_dataset,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(28, 28),
                                                              color_mode='grayscale')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, epochs=4, verbose=1,
                              validation_data=validation_generator)

model.save('models/model.h5')