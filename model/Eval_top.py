import numpy as np
import matplotlib.pyplot as plt

base_path = 'small_32f/'

path_ground_truth = base_path + 'ground_truth.txt'
path_test_file = base_path + '3/HAA500-rgb-i3d-resnet-50-ts-f32/test_1crops_1clips_224.csv'
#path_test_file_numpy = 'small_32f/1/HAA500-rgb-i3d-resnet-50-ts-f32/test_1crops_1clips_224_details.npy'
#result = np.load(path_test_file)


def accuracy(path_ground_truth, path_test_file):
    f = open(path_test_file, 'r')
    lines_test = list()
    for x in f:
        lines_test.append(x.replace('\n',''))

    f = open(path_ground_truth, 'r')
    lines_ground = list()
    for x in f:
        lines_ground.append(x.replace('\n',''))


    top1_good = 0
    top3_good = 0
    top5_good = 0
    for i in range(0,len(lines_test)):
        top_classes = lines_test[i].split(';')
        top3_classes = [top_classes[1], top_classes[2], top_classes[3]]

        if(top_classes[1]) == lines_ground[i]:
            top1_good += 1

        if lines_ground[i] in top_classes:
            top5_good +=1

        if lines_ground[i] in top3_classes:
            top3_good +=1

    return top1_good/(i+1),top3_good/(i+1), top5_good/(i+1)

#print(top1_good/(i+1))
#print(top5_good/(i+1))

