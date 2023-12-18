import tensorflow as tf
from tensorflow import keras
from Data_Load_Preprocessing import *

class DTA_model(tf.keras.Model):
    def __init__(self, drug_vocab, target_vocab, embedding_dim):
        super(DTA_model, self).__init__()
        self.drug_embedding = tf.keras.layers.Embedding(input_dim=len(drug_vocab), output_dim=128)
        self.target_embedding = tf.keras.layers.Embedding(input_dim=len(target_vocab), output_dim=128)

        self.drug_ConvLayers = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 4, activation='relu', padding='valid', strides=1),
            tf.keras.layers.Conv1D(32*2, 6, activation='relu', padding='valid', strides=1),
            tf.keras.layers.Conv1D(32*3, 8, activation='relu', padding='valid', strides=1)])

        self.target_ConvLayers = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 4, activation='relu', padding='valid', strides=1),
            tf.keras.layers.Conv1D(32*2, 8, activation='relu', padding='valid', strides=1),
            tf.keras.layers.Conv1D(32*3, 12, activation='relu', padding='valid', strides=1)])

        self.merge = tf.keras.layers.Concatenate(axis=-1)  # Change axis to 1

        self.FC_layers = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])

    def call(self, inputs):
        Drug_datas, Target_datas = inputs
        x1 = self.drug_embedding(Drug_datas)
        x2 = self.target_embedding(Target_datas)

        x1 = self.drug_ConvLayers(x1)
        x2 = self.target_ConvLayers(x2)

        x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
        x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)

        encoded_interaction = self.merge([x1, x2])
        prediction = self.FC_layers(encoded_interaction)
        return prediction