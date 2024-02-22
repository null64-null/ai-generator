import sys, os
import numpy as np

from util.graph import show_graphs, show_results
from util.progress import show_iter_progress
from util.picture import show_filters

from propagetion.predict import predict, calculate_accuracy
from propagetion.gradient import generate_grads, update_grads

class Supervised_learning:
    def __init__(self, data, layers, error, learning_params, checking_params, isShowProgress, isShowGraph, isShowResult, isShowFilters):
        self.x_train = data['x_train']
        self.t_train = data['t_train']
        self.x_test = data['x_train']
        self.t_test = data['t_train']

        self.layers = layers
        self.error = error

        self.lerning_rate = learning_params['lerning_rate']
        self.batch_size = learning_params['batch_size']
        self.iters = learning_params['iters']

        self.accuracy_check_span = checking_params['accuracy_check_span']
        self.check_mask_size = checking_params['check_mask_size']

        self.errors_x_axis = []
        self.errors = []
        self.accuracy_x_axis = []
        self.accuracy_ratios_train = []
        self.accuracy_ratios_test = []

        self.isShowProgress = isShowProgress
        self.isShowGraph = isShowGraph
        self.isShowResult = isShowResult
        self.isShowFilters = isShowFilters
    
    def learn(self):
        for i in range(self.iters):
            # choose train data from data set
            train_data_length = self.x_train.shape[0]
            batch_mask = np.random.choice(train_data_length, self.batch_size)
            input = self.x_train[batch_mask]
            t = self.t_train[batch_mask]

            # do prediction
            prediction = predict(input, self.layers)

            # calculate error using prediction result
            self.error.generate_error(prediction, t)

            # calculate gradients
            self.error.generate_grad(prediction, t)
            generate_grads(self.layers, self.error)

            # recode results (error), update graph
            self.errors.append(self.error.l)
            self.errors_x_axis.append(i)

            # update parameters in layers
            update_grads(self.layers, self.lerning_rate)

            # accuracy check
            if i % self.accuracy_check_span == 0:
                # get contemporary layer
                test_layers = self.layers

                # get accuracy from train, test data
                accurate_predictions_train, all_data_train, accuracy_ratio_train = calculate_accuracy(self.x_train, self.t_train, test_layers, self.check_mask_size)
                accurate_predictions_test, all_data_test, accuracy_ratio_test = calculate_accuracy(self.x_test, self.t_test, test_layers, self.check_mask_size)
                
                # recode results (accuracy), update graph
                self.accuracy_ratios_train.append(accuracy_ratio_train)
                self.accuracy_ratios_test.append(accuracy_ratio_test)
                self.accuracy_x_axis.append(i)
                
            # indicator iter
            if self.isShowProgress:
                self.show_progress(
                    i,
                    accurate_predictions_train,
                    all_data_train,
                    accuracy_ratio_train,
                    accurate_predictions_test,
                    all_data_test,
                    accuracy_ratio_test
                )

            # show graph
            if self.isShowGraph:
                self.show_progress_graphs()
        
        # show result
        if self.isShowResult:
            self.result()
        if self.isShowFilters:
            self.filters()
    

    def show_progress(
            self,
            i,
            accurate_predictions_train,
            all_data_train,
            accuracy_ratio_train,
            accurate_predictions_test,
            all_data_test,
            accuracy_ratio_test
        ):
        show_iter_progress(
            i+1,
            self.iters,
            self.error.l,
            self.accuracy_check_span,
            accurate_predictions_train,
            all_data_train,
            accuracy_ratio_train,
            accurate_predictions_test,
            all_data_test,
            accuracy_ratio_test,
            self.accuracy_x_axis,
        )
    

    def show_progress_graphs(self):
        show_graphs(
            x1=self.errors_x_axis,
            y1=self.errors,
            x2=self.accuracy_x_axis,
            y2_1=self.accuracy_ratios_train,
            y2_2=self.accuracy_ratios_test,
            x1_label='iter [times]',
            y1_label='error [-]',
            x2_label='iter [times]',
            y2_label='accuracy [%]',
            y1_name='error',
            y2_1_name='accuracy (train)',
            y2_2_name='accuracy (test)',
            title1='error',
            title2='accuracy',
        )


    def result(self):
        show_results(
            x1=self.errors_x_axis,
            y1=self.errors,
            x2=self.accuracy_x_axis,
            y2_1=self.accuracy_ratios_train,
            y2_2=self.accuracy_ratios_test,
            x1_label='iter [times]',
            y1_label='error [-]',
            x2_label='iter [times]',
            y2_label='accuracy [%]',
            y1_name='error',
            y2_1_name='accuracy (train)',
            y2_2_name='accuracy (test)',
            title1='error',
            title2='accuracy',
        )
        
    def filters(self):
        show_filters(self.layers)