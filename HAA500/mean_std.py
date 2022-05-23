import os
import numpy as np
import cv2
import math

def mean_std_com(path):
    my_list = os.listdir(path)
    count = 0

    B = np.zeros(224*224)
    G = np.zeros(224*224)
    R = np.zeros(224*224)

    B_2 = np.zeros(224*224)
    G_2 = np.zeros(224*224)
    R_2 = np.zeros(224*224)

    #for i in range(0,10):
    for i in range(0,len(my_list)):

        images = os.listdir(path + my_list[i])

        for j in range(0,len(images)):

            #img_mean = [0,0,0]
            #img_std = [0,0,0]

            image_test = cv2.imread(path + my_list[i] + '/' + images[j])
            #val = np.reshape(image_test[:, :, 0], -1)
            #img_mean = np.mean(val)
            #img_std = np.std(val)

            image_test = image_test/255.
            B_img_test = np.reshape(image_test[:, :, 0], -1)
            G_img_test = np.reshape(image_test[:, :, 1], -1)
            R_img_test = np.reshape(image_test[:, :, 2], -1)

            B = np.sum([B, B_img_test], axis=0)
            G = np.sum([G, G_img_test], axis=0)
            R = np.sum([R, R_img_test], axis=0)

            B_2 = np.sum([B_2, B_img_test**2], axis=0)
            G_2 = np.sum([G_2, G_img_test**2], axis=0)
            R_2 = np.sum([R_2, R_img_test**2], axis=0)

            count += 1

    R = sum(R)
    G = sum(G)
    B = sum(B)
    R_2 = sum(R_2)
    G_2 = sum(G_2)
    B_2 = sum(B_2)


    psum = np.array([R,G,B])
    psum_sq = np.array([R_2, G_2, B_2])

    cnt = (count * 224 * 224)

    total_mean = psum / cnt
    total_var = (psum_sq/cnt) - (total_mean ** 2)
    total_std = total_var ** 0.5

    return total_mean, total_std




path = 'data_to_train/data/videos/'
mean_fin, std_fin = mean_std_com(path)
print('rgb:')
print('mean: ' + str(mean_fin))
print('std: ' + str(std_fin))
print('-')

path = 'data_to_train_sgm/data/videos/'
mean_fin, std_fin = mean_std_com(path)
print('sgm:')
print('mean: ' + str(mean_fin))
print('std: ' + str(std_fin))
print('-')

path = 'data_to_train_of/opt_flow/videos/'
mean_fin, std_fin = mean_std_com(path)
print('of:')
print('mean: ' + str(mean_fin))
print('std: ' + str(std_fin))



