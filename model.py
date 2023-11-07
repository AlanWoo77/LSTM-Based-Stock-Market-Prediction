class Model(object):
    def __init__(self, df_tweet=None, df_company_tweet=None, df_company=None):
        self.tweet = df_tweet
        self.company_tweet = df_company_tweet
        self.company = df_company

    # define the sentiment extracting method
    def get_sentiment(self):
        self.tweet["post_date"] = pd.to_datetime(self.tweet["post_date"], unit="s")
        self.tweet = self.tweet.merge(
            self.company_tweet, left_on="tweet_id", right_on="tweet_id"
        )
        tsla = self.tweet.loc[tweet["ticker_symbol"] == "TSLA", :].reset_index()

        # create SentimentIntensityAnalyzer object and perform the sentiment analysis
        sia = SentimentIntensityAnalyzer()
        tsla["sentiment_scores"] = tsla["body"].apply(
            lambda tweet: sia.polarity_scores(tweet)
        )
        tsla["compound"] = tsla["sentiment_scores"].apply(
            lambda score_dict: score_dict["compound"]
        )
        tsla["sentiment"] = tsla["compound"].apply(
            lambda c: "positive"
            if c >= 0.05
            else ("neutral" if c > -0.05 and c < 0.05 else "negative")
        )
        tsla["post_date"] = tsla["post_date"].dt.date

        # compute the mean of compounded intraday data
        tsla_sentiment_daily = tsla.groupby("post_date").agg(score=("compound", "mean"))

        return tsla_sentiment_daily

    # train the model for tsla and test
    def tsla_train(self, tsla_daily=None):
        tsla_price_daily = tsla_daily
        start_date = "2015-01-01"
        end_date = "2019-12-31"
        tsla_price_daily = tsla_price_daily.loc[start_date:end_date]
        tsla_price_daily = tsla_price_daily.merge(
            tsla_sentiment_daily, left_index=True, right_index=True
        )

        # prepare the tensor needed for the LSTM training process
        n_timesteps = 30

        features_set = []
        labels = []
        for i in range(n_timesteps, tsla_price_daily.shape[0]):
            features_set.append(tsla_price_daily.iloc[(i - n_timesteps) : i, :])
            labels.append(tsla_price_daily.iloc[i, 3])

        features_set, labels = np.array(features_set), np.array(labels)

        # split the train & test dataset
        n = features_set.shape[0]
        n_train = int(n * 0.80)
        n_test = n - n_train
        # print("n_train = ",str(n_train))
        # print("n_test = ",str(n_test))

        features_set_train, features_set_test = (
            features_set[0:n_train, :, :],
            features_set[n_train:n, :, :],
        )
        labels_train, labels_test = labels[0:n_train], labels[n_train:n]

        # data preprocessing and do the scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Reshape to 2D
        features_set_train_2D = features_set_train.reshape(
            -1, features_set_train.shape[-1]
        )
        features_set_test_2D = features_set_test.reshape(
            -1, features_set_test.shape[-1]
        )

        # Reshape labels to be 2D
        labels_train_2D = labels_train.reshape(-1, 1)
        labels_test_2D = labels_test.reshape(-1, 1)

        # Fit the two scalers
        scaler_X.fit(features_set_train_2D)
        scaler_y.fit(labels_train_2D)

        # Do the transformation
        features_set_train_scaled_2D = scaler_X.transform(features_set_train_2D)
        features_set_test_scaled_2D = scaler_X.transform(features_set_test_2D)

        features_set_train_scaled = features_set_train_scaled_2D.reshape(
            features_set_train.shape
        )
        features_set_test_scaled = features_set_test_scaled_2D.reshape(
            features_set_test.shape
        )

        # Transform the labels and reshape them back
        labels_train_scaled_2D = scaler_y.transform(labels_train_2D)
        labels_test_scaled_2D = scaler_y.transform(labels_test_2D)

        labels_train_scaled = labels_train_scaled_2D.reshape(-1)
        labels_test_scaled = labels_test_scaled_2D.reshape(-1)

        # Set the model
        n_features = 7

        model_lstm = models.Sequential()
        model_lstm.add(
            LSTM(
                512,
                return_sequences=False,
                activation="tanh",
                input_shape=(n_timesteps, n_features),
            )
        )
        model_lstm.add(Dropout(0.4))
        model_lstm.add(layers.Dense(100, activation="linear"))
        model_lstm.add(layers.Dense(1, activation="relu"))

        # train the model
        nepochs = 25
        model_lstm.compile(optimizer="adam", loss="mse")
        history = model_lstm.fit(
            features_set_train_scaled,
            labels_train_scaled,
            epochs=nepochs,
            batch_size=128,
            validation_data=(features_set_test_scaled, labels_test_scaled),
        )
        return model_lstm, history

        # create the plot method
        def plot(self, history_lstm=None):
            # define all the plot functions
            plt.clf()  # clear figure
            history_lstm = history.history
            train_mse = history_lstm["loss"]
            test_mse = history_lstm["val_loss"]
            min_test_mse = min(test_mse)

            plt.figure(figsize=(16, 6))
            plt.plot(
                range(1, nepochs + 1), train_mse, "b", label="train MSE", color="blue"
            )
            plt.plot(
                range(1, nepochs + 1), test_mse, "b", label="test MSE", color="red"
            )

            # plt.ylim((0, 0.0001))
            plt.title(
                "LSTM: min(test MSE) = " + str(round(min(test_mse), 6)), fontsize=16
            )
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("MSE", fontsize=16)
            plt.legend(loc="upper right", fontsize=16)
            # plt.ylim(0,0.0015)
            plt.show()

            # do the plot of the prediction
            train = tsla_price_daily.iloc[30 : (30 + len(labels_train)), :]
            y_hat_train = model_lstm.predict(features_set_train_scaled)
            y_hat_train = scaler_y.inverse_transform(y_hat_train)
            train["prediction"] = y_hat_train

            valid = tsla_price_daily.iloc[(30 + len(labels_train)) :, :]
            y_hat_valid = model_lstm.predict(features_set_test_scaled)
            y_hat_valid = scaler_y.inverse_transform(y_hat_valid)
            valid["prediction"] = y_hat_valid
            plt.style.use("seaborn")
            plt.figure(figsize=(16, 6))
            plt.plot(train["Close"], label="Train")
            plt.plot(valid[["Close", "prediction"]], label=["Validation", "Prediction"])
            plt.title(
                "Tesla With Public Opinion (min(mse) = {:.4f})".format(min_test_mse)
            )
            plt.legend()
            plt.show()

        return None

        # train the model for etfs
        def etfs_train(self, path=None):
            models = []
            for path in paths:
                price_daily = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                regex = r".*/(.*).csv"
                company = re.findall(regex, path)[0]

                # prepare the tensor needed for the LSTM training process
                n_timesteps = 30

                features_set = []
                labels = []
                for i in range(n_timesteps, price_daily.shape[0]):
                    features_set.append(price_daily.iloc[(i - n_timesteps) : i, :])
                    labels.append(price_daily.iloc[i, 3])

                features_set, labels = np.array(features_set), np.array(labels)

                # split the train & test dataset
                n = features_set.shape[0]
                n_train = int(n * 0.80)
                n_test = n - n_train
                # print("n_train = ",str(n_train))
                # print("n_test = ",str(n_test))

                features_set_train, features_set_test = (
                    features_set[0:n_train, :, :],
                    features_set[n_train:n, :, :],
                )
                labels_train, labels_test = labels[0:n_train], labels[n_train:n]

                # data preprocessing and do the scaling
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                # Reshape to 2D
                features_set_train_2D = features_set_train.reshape(
                    -1, features_set_train.shape[-1]
                )
                features_set_test_2D = features_set_test.reshape(
                    -1, features_set_test.shape[-1]
                )

                # Reshape labels to be 2D
                labels_train_2D = labels_train.reshape(-1, 1)
                labels_test_2D = labels_test.reshape(-1, 1)

                # Fit the two scalers
                scaler_X.fit(features_set_train_2D)
                scaler_y.fit(labels_train_2D)

                # Do the transformation
                features_set_train_scaled_2D = scaler_X.transform(features_set_train_2D)
                features_set_test_scaled_2D = scaler_X.transform(features_set_test_2D)

                features_set_train_scaled = features_set_train_scaled_2D.reshape(
                    features_set_train.shape
                )
                features_set_test_scaled = features_set_test_scaled_2D.reshape(
                    features_set_test.shape
                )

                # Transform the labels and reshape them back
                labels_train_scaled_2D = scaler_y.transform(labels_train_2D)
                labels_test_scaled_2D = scaler_y.transform(labels_test_2D)

                labels_train_scaled = labels_train_scaled_2D.reshape(-1)
                labels_test_scaled = labels_test_scaled_2D.reshape(-1)

                # Set the model
                n_features = 6

                model_lstm = models.Sequential()
                model_lstm.add(
                    LSTM(
                        512,
                        return_sequences=False,
                        activation="tanh",
                        input_shape=(n_timesteps, n_features),
                    )
                )
                model_lstm.add(layers.Dense(100, activation="linear"))
                model_lstm.add(Dropout(0.4))
                model_lstm.add(layers.Dense(1, activation="relu"))

                # train the model
                nepochs = 15
                model_lstm.compile(optimizer="adam", loss="mse")
                history = model_lstm.fit(
                    features_set_train_scaled,
                    labels_train_scaled,
                    epochs=nepochs,
                    batch_size=128,
                    validation_data=(features_set_test_scaled, labels_test_scaled),
                )

                models.append(model_lstm)
                self.plot(history_lstm=history)

            return models
