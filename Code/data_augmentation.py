import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2 as cv

##########################################################--function--###################################################################
def visualize(original,augmented,type):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title(f'{type} image')
    plt.imshow(augmented)
    plt.show()
    
    #Conversion en nuance de gris
def conv_niv_gris(nom_image,image_save):
    (l,h) = nom_image.size
    image_g = Image.new('RGB',(l,h))
    for x in range(l):
        for y in range(h):
            r,g,b = nom_image.getpixel((x,y))
            value = int(r*299/1000 + g*587/1000 + b*114/1000) #sensibilite de l'oeil humain a chaque couleur
            p = (value, value, value) #ou p = value
            image_g.putpixel((x,y),p)
    image_g.save("image/"+str(image_save)+".jpeg")
    image_g.show()

##########################################################--Paths--#####################################################################
training_path = 'F:/Ecole/GE5A/Imagerie_num/Photos_train/'

##########################################################--Directory list--############################################################
os.chdir(training_path)					#change the current working directory
#print(os.getcwd())						#check it
dir_list = os.listdir(training_path) 	#List the directory in training_path
#print(dir_list)						#check it
length_dir=len(dir_list)				#Nb of directory

##########################################################--Processing--#################################################################
for i in range(length_dir):
    new_directory = training_path+dir_list[i]
    os.chdir(new_directory)
    print(os.getcwd())
    image_list = os.listdir(new_directory)
    #print(image_list)
    length_ima=len(image_list)
    #print(length_ima)
    for j in range(length_ima):
        #print(image_list[j])
        #Load image
        image = tf.keras.utils.load_img(image_list[j])
        
        #DATA AUGMENTATION TENSORFLOW
        flipped_lr = tf.image.flip_left_right(image)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_flipped.jpg",flipped_lr)
        flipped_ud = tf.image.flip_up_down(image)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_updown.jpg",flipped_ud)
        transpose = tf.image.transpose(image)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_transpose.jpg",transpose)
        rotated = tf.image.rot90(image)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_rotate.jpg",rotated)
        saturated = tf.image.adjust_saturation(image, 3)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_saturate.jpg",saturated)
        gamma = tf.image.adjust_gamma(image, 2, 4)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_gamma.jpg",gamma)
        quality = tf.image.adjust_jpeg_quality(image, 25)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_quality.jpg",quality)
        brightness = tf.image.adjust_brightness(image,0.4)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_brightness.jpg",brightness)
        contrast = tf.image.adjust_contrast(image, 0.6)
        tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_contrast.jpg",contrast)

        
                
                

##########################################################--TEST--#################################################################
# flipped_lr = tf.image.flip_left_right(image)
# visualize(image, flipped_lr,"flipped")
# 
# flipped_ud = tf.image.flip_up_down(image)
# visualize(image, flipped_ud,"flipped")
# 
# transpose = tf.image.transpose(image)
# visualize(image, transpose,"transpose")
# 
# rotated = tf.image.rot90(image)
# visualize(image, rotated,"rotated")

# saturated = tf.image.adjust_saturation(image, 3)
# visualize(image, saturated,"saturated")
# 
# for i in range(2,6):
#     for j in range(2,6):
#         saturated = tf.image.adjust_gamma(image, i, j)
#         visualize(image, saturated,"gamma")
# 
# for i in np.arange(-0.8,0.9,0.3):
#     saturated = tf.image.adjust_hue(image, i)
#     visualize(image, saturated,"hue")
# #
# for i in range (10,100,10):
#     saturated = tf.image.adjust_jpeg_quality(image, i)
#     visualize(image, saturated,"quality")
# 
# 
# #Modication aleatoire de luminosite
# for i in range(3):
#     seed = (i, 0)  # tuple of size (2,)
#     stateless_random_brightness = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)
#     visualize(image, stateless_random_brightness,"random brightness")
# 
# #Modication aleatoire de contraste
# for i in range(3):
#     seed = (i, 0)  # tuple of size (2,)
#     stateless_random_contrast = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
#     visualize(image, stateless_random_contrast,"ramdom contrast")
#     
# cropped = tf.image.central_crop(image, central_fraction=0.5)
# visualize(image,cropped,"cropped")

# for x in range(2,6):
#     for y in range(2,6):
#         gamma = tf.image.adjust_gamma(image, x, y)
#         tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_"+str(x)+str(y)+"_gamma.jpg",gamma)
# for x in range (10,100,10):
#     quality = tf.image.adjust_jpeg_quality(image, x)
#     tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_"+str(x)+"_quality.jpg",quality)
# for x in range(3):
#     seed = (x, 0)  # tuple of size (2,)
#     stateless_random_brightness = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)
#     tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_"+str(x)+"_r_brightness.jpg",stateless_random_brightness)
# for x in range(3):
#     seed = (x, 0)  # tuple of size (2,)
#     stateless_random_contrast = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
#     tf.keras.utils.save_img(str(dir_list[i])+"_"+str(j)+"_"+str(x)+"_r_contrast.jpg",stateless_random_contrast)
