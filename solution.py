from flame.schemas.star import StarModel, StarAnalyzer, StarAggregator

from io import StringIO

import numpy as np
import pandas as pd
from keras.src.backend import shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import StandardScaler


class MyAnalyzer(StarAnalyzer):
    def __init__(self, flame):
        super().__init__(flame)

    def analysis_method(self, data, aggregator_results):
        x_train, y_train = self.load_split_boston(data)
        x_test, y_test = self.load_test_split_boston(data)
        # Load the split data
        if aggregator_results is None:
            model = self.create_model()
        else:
            model = aggregator_results
        self.model = self.train_model(x_train, y_train, model)
        self.evaluate_model(x_test, y_test, model)
        return model


    def load_test_split_boston(self,data):
        uc_store = data[0]
        test_data_name = [name for name in uc_store.keys() if name.startswith('test_split_')][0]
        test_data  = pd.read_csv(StringIO(uc_store[test_data_name]))
        # print test data shape
        print(f'test data shape {test_data.shape}')
        x_test = test_data.drop(columns=['target']).values
        y_test = test_data['target'].values

        return x_test, y_test

    def load_split_boston(self, data):
        uc_store = data[0]
        train_data_name = [name for name in uc_store.keys() if name.startswith('train_split_')][0]
        train_data = pd.read_csv(StringIO(uc_store[train_data_name]))
        print(f'test data shape {train_data.shape}')
        x_train = train_data.drop(columns=['target']).values
        y_train = train_data['target'].values

        return x_train, y_train

    def evaluate_model(self, x_test, y_test, model):
        # Normalize the data
        scaler = StandardScaler()
        x_test = scaler.fit_transform(x_test)

        # Evaluate the model
        loss, mae = model.evaluate(x_test, y_test, verbose=0)

        print(f'Test - Loss: {loss}, MAE: {mae}')


    def train_model(self,x_split, y_split , model):

        # Normalize the data
        scaler = StandardScaler()
        x_split = scaler.fit_transform(x_split)
        # Train the model for one epoch
        model.fit(x_split, y_split, epochs=1, batch_size=32)

        return model

    def create_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(13,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


class MyAggregator(StarAggregator):
    def __init__(self, flame):
        super().__init__(flame)

    def aggregation_method(self, analysis_results):
        # print type of analysis_results
        print(type(analysis_results[0]))
        return self.average_weights(analysis_results)

    def has_converged(self, result, last_result,num_iterations):
        if num_iterations >= 10:
            return True
        else:
            return False

    def average_weights(self, models):
        # Get the weights from each model
        weights = [model.get_weights() for model in models]

        # Average the weights
        avg_weights = []
        for weights_list_tuple in zip(*weights):
            avg_weights.append(np.mean(np.array(weights_list_tuple), axis=0))

        return avg_weights

def main():
    StarModel(analyzer=MyAnalyzer,
              aggregator=MyAggregator,
              data_type='s3',
              output_type='pickle',
              simple_analysis=False)


if __name__ == "__main__":
    main()
