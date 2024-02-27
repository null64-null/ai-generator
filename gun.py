import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from util.progress import show_iter_progress_for_gun
from util.picture import show_generated_pictures
from propagetion.predict import predict
from propagetion.gradient import generate_grads, update_grads

class Gun:
    def __init__(self, data, batch_size, generator_layers, discriminator_layers, error, learning_params, checking_params, is_show_progress, is_show_pictures, is_show_result):
        x_auth = data['x_auth']
        t_auth = data['t_auth']

        self.x_auth = x_auth
        self.t_auth = t_auth

        self.batch_size = batch_size
        self.labels_number = t_auth.shape[1]
       
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.error = error

        self.lerning_rate = learning_params['lerning_rate']
        self.iters = learning_params['iters']

        self.picture_check_span = checking_params['picture_check_span']

        self.is_show_progress = is_show_progress
        self.is_show_graph = is_show_pictures
        self.is_show_result = is_show_result
    
    def learn(self):
        # make auth, fake t
        auth = np.zeros(self.batch_size) # 0 "auth"
        fake = np.ones(self.batch_size) # 1 "fake"

        test_input = np.array([])

        for i in range(self.iters):
            # choose auth images (number = batch size)
            batch_mask = np.random.choice(self.x_auth.shape[0], self.batch_size)
            x_auth = self.x_auth[batch_mask]

            ################# [ generator の更新 ] generate が作成した画像が 0 "true" になるようにする ###############
            # make random input
            input = np.random.uniform(low=0, high=1, size=(self.batch_size, self.labels_number))
            # do generation and discrimination
            generated_pictures = predict(input, self.generator_layers)
            discriminations = predict(generated_pictures, self.discriminator_layers)

            # judge (t = 0 "true")
            self.error.generate_error(discriminations, auth)

            # calculate gradients (only generator_layer)
            self.error.generate_grad(discriminations, auth)
            generate_grads(self.discriminator_layers, self.error)
            generate_grads(self.generator_layers, self.discriminator_layers[0])

            # update params (only generator_layer)
            update_grads(self.generator_layers, self.lerning_rate)

            # error reset
            self.error.initialize()


            ######## [ discriminator の更新 ] discriminator が generator の画像を 1 "fake", 本物の画像を 0 "auth"  にできるようにする ########
            # make images by updated generator
            input = np.random.uniform(low=0, high=1, size=(self.batch_size, self.labels_number))
            generated_pictures = predict(input, self.generator_layers)

            # make joint image (auth & generated pictures)
            pictures = np.concatenate((x_auth, generated_pictures), axis=0)
            t = np.concatenate((auth, fake), axis=0)

            # discriminate
            discriminations = predict(pictures, self.discriminator_layers)

            # make error (by t)
            self.error.generate_error(discriminations, t)

            # calculate gradients (only discriminator_layer)
            self.error.generate_grad(discriminations, t)
            generate_grads(self.discriminator_layers, self.error) ####

            # update params (only discriminator_layer)
            update_grads(self.discriminator_layers, self.lerning_rate) ####
            
            # error reset
            self.error.initialize()

            ############################################################################################

            # picture check
            if i % self.picture_check_span == 0:
                test_generator = self.generator_layers
                test_input = np.random.uniform(low=0, high=1, size=(10, self.labels_number))
                test_pictures = predict(test_input, test_generator)
                show_generated_pictures(test_pictures)
                
            # indicator iter
            if self.is_show_progress:
                show_iter_progress_for_gun(i+1, self.iters, self.picture_check_span, test_input)
        
        # show result
        if self.is_show_result:
            test_generator = self.generator_layers
            test_input = np.random.uniform(low=0, high=1, size=(10, self.labels_number))
            test_pictures = predict(test_input, test_generator)
            show_generated_pictures(test_pictures)

                