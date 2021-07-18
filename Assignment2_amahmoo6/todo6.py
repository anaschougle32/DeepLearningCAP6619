class Todo6_A:
    def __init__(self):
        self.build()

    def build(self):
        self.load_data()
        self.normalize_data()
        self.compile_model()
        self.validation_log_each_fold()
        self.plot_A()
        self.plot_B()
        self.train_final_model()

    def load_data(self):
        from tensorflow.keras.datasets import boston_housing
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = boston_housing.load_data()
        print(self.train_data.shape)
        print(self.test_data.shape)
        print(self.train_targets)

    def normalize_data(self):
        self.mean = self.train_data.mean(axis=0)
        self.train_data -= self.mean
        self.std = self.train_data.std(axis=0)
        self.train_data /= self.std
        self.test_data -= self.mean
        self.test_data /= self.std

    def build_model(self):
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return self.model

    def compile_model(self):
        self.k = 4
        self.num_val_samples = len(self.train_data) // k
        self.num_epochs = 100
        self.all_scores = []
        for i in range(k):
            print(f"Processing fold #{i}")
            val_data = self.train_data[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            val_targets = self.train_targets[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            self. partial_train_data = np.concatenate(
                [self.train_data[:i * self.num_val_samples],
                 self.train_data[(i + 1) * self.num_val_samples:]],
                axis=0)
            self.partial_train_targets = np.concatenate(
                [self.train_targets[:i * self.num_val_samples],
                 self.train_targets[(i + 1) * self.num_val_samples:]],
                axis=0)
            model = self.build_model()
            model.fit(self.partial_train_data, self.partial_train_targets,
                      epochs=self.num_epochs, batch_size=16, verbose=0)
            self.val_mse, self.val_mae = model.evaluate(val_data, val_targets, verbose=0)
            self.all_scores.append(self.val_mae)

    def validation_log_each_fold(self):
        print(self.all_scores)
        print(np.mean(self.all_scores))
        self.num_epochs = 500
        self.all_mae_histories = []
        for i in range(self.k):
            print(f"Processing fold #{i}")
            self.val_data = self.train_data[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            self.val_targets = self.train_targets[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            self.partial_train_data = np.concatenate(
                [self.train_data[:i * self.num_val_samples],
                 self.train_data[(i + 1) * self.num_val_samples:]],
                axis=0)
            self.partial_train_targets = self.np.concatenate(
                [self.train_targets[:i * self.num_val_samples],
                 self.train_targets[(i + 1) * self.num_val_samples:]],
                axis=0)
            model = self.build_model()
            history = model.fit(self.partial_train_data, self.partial_train_targets,
                                validation_data=(self.val_data, self.val_targets),
                                epochs=self.num_epochs, batch_size=16, verbose=0)
            self.mae_history = history.history["val_mae"]
            self.all_mae_histories.append(self.mae_history)
        self.average_mae_history = [
            np.mean([x[i] for x in self.all_mae_histories]) for i in range(self.num_epochs)]


    def plot_A(self):
        plt.plot(range(1, len(self.average_mae_history) + 1), self.average_mae_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()
    def plot_B(self):
        self.truncated_mae_history = self.average_mae_history[10:]
        plt.plot(range(1, len(self.truncated_mae_history) + 1), self.truncated_mae_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def train_final_model(self):
        model = self.build_model()
        model.fit(self.train_data, self.train_targets,
                  epochs=130, batch_size=16, verbose=0)
        self.test_mse_score, self.test_mae_score = model.evaluate(self.test_data, self.test_targets)
        print(self.test_mae_score)
        predictions = model.predict(self.test_data)
        predictions[0]