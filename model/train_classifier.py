import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import joblib
from utl import get_classes, parse_csv, get_accuracy
'''
def get_accuracy(dec,gt):
    top1_good = 0
    top3_good = 0
    top5_good = 0
    for i in range(0, len(dec)):

        H = dec[i]
        ind = np.argpartition(H, -5)[-5:]
        values = H[ind]
        x = np.argsort(values)
        ind = ind[x]
        ind = np.flip(ind)

        top3_classes = [ind[0], ind[1], ind[2]]

        if gt[i] == ind[0]:
            top1_good += 1
        if gt[i] in top3_classes:
            top3_good += 1
        if gt[i] in ind:
            top5_good += 1

    return top1_good / (i + 1), top3_good / (i + 1), top5_good / (i + 1)

def get_classes(path):
    f = open(path, 'r')
    lines_ground = list()
    for x in f:
        lines_ground.append(int(x.replace('\n', '')))

    return lines_ground

def parse_csv(path):
    file = open(path)
    csvreader = csv.reader(file)
    header = next(csvreader)
    #print(header)
    rows = []
    for row in csvreader:
        rows.append(row)
    #print(rows)
    file.close()

    header = list(np.float_(header))
    rows = list(np.float_(rows))

    return header,rows

'''

path_rgb = 'data/rgb/softmax_values.csv'
path_sgm = 'data/sgm/softmax_values.csv'
path_of = 'data/of/softmax_values.csv'

path_gt = 'data/gt_val.txt'
gt = get_classes(path_gt)

header,rows_rgb = parse_csv(path_rgb)
_,rows_sgm = parse_csv(path_sgm)
_,rows_of = parse_csv(path_of)

train_data = []
for i in range(0,len(rows_rgb)):
    H = np.concatenate((rows_rgb[i],rows_sgm[i],rows_sgm[i]))
    train_data.append(H)


path_rgb = 'data_train/rgb/softmax_values.csv'
path_sgm = 'data_train/sgm/softmax_values.csv'
path_of = 'data_train/of/softmax_values.csv'

path_gt = 'data_train/gt_train.txt'
gt_train = get_classes(path_gt)
for i in range(0, len(gt_train)):
    gt.append(gt_train[i])

header,rows_rgb = parse_csv(path_rgb)
_,rows_sgm = parse_csv(path_sgm)
_,rows_of = parse_csv(path_of)

#train_data = []
for i in range(0,len(rows_rgb)):
    H = np.concatenate((rows_rgb[i],rows_sgm[i],rows_sgm[i]))
    train_data.append(H)

le = LabelEncoder()
train_labels = le.fit_transform(gt)

classif = 0

if classif == 0:
    model = svm.LinearSVC(C=1, max_iter=100,tol=1e-3)
elif classif == 1:
    model = svm.LinearSVC( C=0.5,max_iter=100,tol=1e-3)
elif classif == 2:
    model = svm.LinearSVC( C=0.1,max_iter=100,tol=1e-3)
elif classif == 3:
    model = KNeighborsClassifier(n_neighbors=3)


model.fit(train_data,train_labels)

model_path = 'classif_m/modelSVM.npy'
joblib.dump(model, model_path)




