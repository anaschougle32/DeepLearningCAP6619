class Todo5_A:

    # init method or constructor
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y
        self.build()

    def build(self):
        self.build_model()
        self.compile_model()
        self.train_data()
        self.train_Model()
        self.plot_A()
        self.plot_B()

    def build_model(self):
        print("build model")
        self.model = keras.Sequential([
            layers.Dense(48, activation="softplus"),
            layers.Dense(48, activation="softplus"),
            layers.Dense(64, activation="softplus"),
            layers.Dense(64, activation="softplus"),
            layers.Dense(46, activation="softmax")
        ])

    def compile_model(self):
        print("compile model")
        self.model.compile(optimizer="rmsprop",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def train_data(self):
        print("train data")
        self.x_val_optional = self.x_train[:1000]
        self.partial_x_train_optional = self.x_train[1000:]
        self.y_val_optional = self.y_train[:1000]
        self.partial_y_train_optional = self.y_train[1000:]

    def train_Model(self):
        print("train model")
        self.history_optional = self.model.fit(self.partial_x_train_optional,
                                               self.partial_y_train_optional,
                                               epochs=20,
                                               batch_size=512,
                                               validation_data=(self.x_val_optional, self.y_val_optional))

    def plot_A(self):
        print("Plotting the training and validation loss")
        self.loss = self.history_optional.history["loss"]
        self.val_loss = self.history_optional.history["val_loss"]
        self.epochs = range(1, len(self.loss) + 1)
        plt.plot(self.epochs, self.loss, "bo", label="Training loss")
        plt.plot(self.epochs, self.val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_B(self):
        print("Plotting the training and validation loss")
        plt.clf()
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, acc, "bo", label="Training accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


Todo5_A(x_train, y_train)