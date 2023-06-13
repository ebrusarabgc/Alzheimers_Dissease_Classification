import tensorflow as tf


def create_model(input_shape, base_model, num_classes):
    base_model.trainable = True

    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def compile_model(model, learning_rate):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )


def train_model(model, train_data, valid_data, epochs):
    model.fit(train_data, validation_data=valid_data, epochs=epochs, verbose=False)


def evaluate_model(model, valid_data):
    model.evaluate(valid_data)
