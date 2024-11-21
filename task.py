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
        #TODO split the data into x_test and y_test

        #return x_test, y_test

    def load_split_boston(self, data):
        uc_store = data[0]
        train_data_name = [name for name in uc_store.keys() if name.startswith('train_split_')][0]
        train_data = pd.read_csv(StringIO(uc_store[train_data_name]))
        #TODO split the data into x_train and y_train

        #return x_train, y_train

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
        # TODO create a model using  tensorflow.keras.models.Sequential
        pass


class MyAggregator(StarAggregator):
    def __init__(self, flame):
        super().__init__(flame)

    def aggregation_method(self, analysis_results):
        # print type of analysis_results
        print(type(analysis_results[0]))
        return self.aggregate_weights(analysis_results)

    def has_converged(self, result, last_result,num_iterations):
        if num_iterations >= 10:
            return True
        else:
            return False

    def aggregate_weights(self, models):
        #TODO aggregate the weights of the models

        return avg_weights

def main():
    StarModel(analyzer=MyAnalyzer,
              aggregator=MyAggregator,
              data_type='s3',
              output_type='pickle',
              simple_analysis= False)


if __name__ == "__main__":
    main()
