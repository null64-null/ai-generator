import sys, os
import numpy as np
import matplotlib.pyplot as plt

from util.progress import show_iter_progress_for_gun

from propagetion.predict import predict
from propagetion.gradient import generate_grads, update_grads

class Gun:
    def __init__(self, data, generator_layers, discriminator_layers, error, learning_params, checking_params, is_show_progress, is_show_pictures, is_show_result):
        self.x_auth = np.array([])

        x_auth = data['x_auth']
        t_auth = data['t_auth']

        self.labels_number = t_auth.shape[1]
        for i in range(self.labels_number):
            mask = np.where(t_auth[:, i] == 1)[0]
            self.x_auth.append(x_auth[mask])

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
        for i in range(self.iters):
            # choose train data from data set
            input = np.eye(10)
            x_auth = np.array([])
            for i in range(self.x_auth[0]):
                x_auth.append(self.x_auth[i], 1)
            auth = np.zeros(self.labels_number).T # 0 "auth"
            fake = np.ones(self.labels_number).T # 1 "fake"


            ################# generate が作成した画像が 0 "true" になるように generator を更新 ###############
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


            ######## discriminator が generator の画像を 1 "fake" にできるように discriminator を更新 ########
            # calculate gradients (only discriminator_layer)
            self.error.generate_grad(discriminations, fake)
            generate_grads(self.discriminator_layers, self.error)

            # update params (only discriminator_layer)
            update_grads(self.discriminator_layers, self.lerning_rate)
            
            # error reset
            self.error.initialize()


            ########### discriminator が 本物の画像を 0 "auth" にできるように discriminator を再更新 ##########
            # do discrimination
            discriminations = predict(x_auth, self.discriminator_layers)

            # judge (t = 0 "auth")
            self.error.generate_error(discriminations, auth)

            # calculate gradients (only discriminator_layer)
            self.error.generate_grad(discriminations, auth)
            generate_grads(self.discriminator_layers, self.error)

            # update params (only discriminator_layer)
            update_grads(self.discriminator_layers, self.lerning_rate)

            # error reset
            self.error.initialize()
            ############################################################################################


            # picture check
            if i % self.picture_check_span == 0:
                generated_pictures = predict(input, self.generator_layers)
                n, c, _, _ = generated_pictures.shape

                _, axes = plt.subplots(1, n, figsize=(n * 2, 2))
                
                for i in range(n):
                    if c == 1:
                        axes[i].imshow(generated_pictures[i, 0], cmap='gray')
                        axes[i].axis('off')
                    else:
                        axes[i].imshow(np.transpose(generated_pictures[i], (1, 2, 0)))  
                        axes[i].axis('off') 

                plt.tight_layout()
                plt.show()

                
            # indicator iter
            if self.is_show_progress:
                show_iter_progress_for_gun(i, self.iters, self.picture_check_span)
        
        # show result
        if self.is_show_result:
            generated_pictures = predict(input, self.generator_layers)
            n, c, _, _ = generated_pictures.shape

            _, axes = plt.subplots(1, n, figsize=(n * 2, 2))
                    
            for i in range(n):
                if c == 1:
                    axes[i].imshow(generated_pictures[i, 0], cmap='gray')
                    axes[i].axis('off')
                else:
                    axes[i].imshow(np.transpose(generated_pictures[i], (1, 2, 0)))  
                    axes[i].axis('off') 

                plt.tight_layout()
                plt.show()