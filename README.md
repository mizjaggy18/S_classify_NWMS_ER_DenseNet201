S_classify_NWMS_ER_DenseNet201
# S_classify_NWMS_ER_DenseNet201

Trained using 3333 nuclei based on validation by collaborative pathologists.

DenseNet201 model layers:

def densemodel():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ])
    tensor = tf.keras.Input((224, 224, 3))
    x = tf.cast(tensor, tf.float32)
    x = tf.keras.applications.densenet.preprocess_input(
        x, data_format=None)
    x = data_augmentation(x)
    x = pretrained_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(4)(x)
    x = tf.nn.softmax(x)
    model = tf.keras.Model(tensor, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
    
