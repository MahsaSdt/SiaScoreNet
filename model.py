import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import backend as K

def SiaScoreNet():
    def swish(x):
        return x * tf.nn.sigmoid(x)

    dropout_rate = 0.2
    ac = swish

    input_pep = Input(shape=(320,), name='peptide')
    x_pep = layers.Dense(256, activation=ac)(input_pep)
    x_pep = layers.BatchNormalization()(x_pep)
    x_pep = layers.Dropout(dropout_rate)(x_pep)
    x_pep = layers.Dense(128, activation=ac)(x_pep)
    x_pep = layers.BatchNormalization()(x_pep)
    x_pep = layers.Dropout(dropout_rate)(x_pep)
    x_pep = layers.Dense(64, activation=ac)(x_pep)

    input_hla = Input(shape=(320,), name='hla')
    x_hla = layers.Dense(256, activation=ac)(input_hla)
    x_hla = layers.BatchNormalization()(x_hla)
    x_hla = layers.Dropout(dropout_rate)(x_hla)
    x_hla = layers.Dense(128, activation=ac)(x_hla)
    x_hla = layers.BatchNormalization()(x_hla)
    x_hla = layers.Dropout(dropout_rate)(x_hla)
    x_hla = layers.Dense(64, activation=ac)(x_hla)

    x1 = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([x_hla, x_pep])
    x2 = layers.Multiply()([x_pep, x_hla])
    x = layers.Concatenate()([x1, x2])

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation=ac)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation=ac)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(16, activation=ac)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(8, activation=ac)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation=ac)(x)

    input_scores = Input(shape=(9,), name='scores')
    x = layers.Concatenate()([x, input_scores])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_pep, input_hla, input_scores], outputs=output)

    try:
        from tensorflow.keras.optimizers.experimental import AdamW
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
    except ImportError:
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='AUC'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.AUC(name='AUPR', curve='PR')
        ]
    )

    return model
