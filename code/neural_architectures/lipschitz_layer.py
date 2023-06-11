import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Layer, InputLayer


# Hidden to hidden matrices A,W with trainable MA i MW
# Input to hidden
# Output to hidden D
# Offset b

class LipschitzCell(Layer):
    def __init__(self, hiddenDimension, inputDimension, outputDimension, betaA, gammaA, betaW, gamaW, offsetB,
                 timeStep):
        super().__init__()

        self.betaA = betaA
        self.gamaA = gammaA
        self.betaW = betaW
        self.gamaW = gamaW
        self.offsetB = offsetB
        self.timeStep = timeStep
        self.Eye = tf.eye(hiddenDimension, hiddenDimension)
        self.HiddenState = tf.zeros(shape=(hiddenDimension, 1))

        M_A_init = tf.random_uniform_initializer()
        self.M_A = tf.Variable(
            initial_value=M_A_init(shape=(hiddenDimension, hiddenDimension), dtype=float),
            trainable=True,
        )

        M_W_init = tf.random_uniform_initializer()
        self.M_W = tf.Variable(
            initial_value=M_W_init(shape=(hiddenDimension, hiddenDimension), dtype=float),
            trainable=True,
        )

        U_init = tf.random_uniform_initializer()
        self.U = tf.Variable(
            initial_value=U_init(shape=(hiddenDimension, inputDimension), dtype=float),
            trainable=True,
        )

        D_init = tf.random_uniform_initializer()
        self.D = tf.Variable(
            initial_value=D_init(shape=(hiddenDimension, outputDimension), dtype=float),
            trainable=True,
        )

    def call(self, inputs):
        A = (1 - self.betaA) * (self.M_A + tf.transpose(self.M_A)) + \
            self.betaA * (self.M_A + tf.transpose(self.M_A)) - self.gamaA * self.Eye
        W = (1 - self.betaW) * (self.M_A + tf.transpose(self.M_A)) + \
            self.betaW * (self.M_A + tf.transpose(self.M_A)) - self.gamaW * self.Eye

        self.HiddenState = self.HiddenState + self.timeStep * A + self.timeStep * tf.math.tanh(
            W * self.HiddenState + self.U * inputs + self.offsetB)
        return self.D * self.HiddenState


if __name__ == "__main__":
    layer = LipschitzCell(64, 1, 1, 0.75, 1, 0.65, 1, 1, 1)
    x = 17.1
    y = layer(x)
    print(y)
    model = Sequential()
    model.add(InputLayer(input_shape=1))
    model.add(LipschitzCell(64, 1, 1, 0.75, 1, 0.65, 1, 1, 1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #model.fit()



