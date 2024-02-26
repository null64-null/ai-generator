import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from util.progress import show_iter_progress_for_gun

from propagetion.predict import predict
from propagetion.gradient import generate_grads, update_grads

class Gun:
    def __init__(self, data, labels_number, generator_layers, discriminator_layers, error, learning_params, checking_params, is_show_progress, is_show_pictures, is_show_result):
        x_auth = data['x_auth']
        t_auth = data['t_auth']

        self.x_auth = x_auth
        self.t_auth = t_auth
        self.labels_number = labels_number
       
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
        a = []
        for i in range(self.iters):
            a.append(i)
            print(a)
            _, c, h, w = self.x_auth.shape

            # make input: expected output 0~9
            input = np.eye(10)

            # choose auth images 0~9
            x_auth = np.zeros((10, c, h, w))
            for i in range(10):
                random_row_index = np.random.choice(np.where(self.t_auth[:, i] == 1)[0])
                x_auth[i] = self.x_auth[random_row_index]

            # make auth, fake t
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
            # judge (t = 1 "fake")
            self.error.generate_error(discriminations, fake)

            # calculate gradients (only discriminator_layer)
            self.error.generate_grad(discriminations, fake)
            generate_grads(self.discriminator_layers, self.error) ####

            # update params (only discriminator_layer)
            update_grads(self.discriminator_layers, self.lerning_rate) ####
            
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
                n, c, h, w = generated_pictures.shape
                
                images_combined = []

                for i in range(n):
                    image = generated_pictures[i, :, :, :].reshape(c, h, w)
                    image = image.reshape(h, w)

                    min_val = np.min(image)
                    max_val = np.max(image)

                    image_scaled = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                    images_combined.append(image_scaled)

                    if i < n - 1:
                        images_combined.append(np.zeros((h, round(w / 2)), dtype=np.uint8))
                
                combined_image = np.hstack(images_combined)
                combined_image = Image.fromarray(combined_image)
                combined_image.show()
                
            # indicator iter
            if self.is_show_progress:
                show_iter_progress_for_gun(i, self.iters, self.picture_check_span)
        
        # show result
        if self.is_show_result:
            generated_pictures = predict(input, self.generator_layers)
            n, c, h, w = generated_pictures.shape
            
            images_combined = []

            for i in range(n):
                image = generated_pictures[i, :, :, :].reshape(c, h, w)
                image = image.reshape(h, w)

                min_val = np.min(image)
                max_val = np.max(image)

                image_scaled = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                images_combined.append(image_scaled)

                if i < n - 1:
                    images_combined.append(np.zeros((h, round(w / 2)), dtype=np.uint8))
            
            combined_image = np.hstack(images_combined)
            combined_image = Image.fromarray(combined_image)
            combined_image.show()
                