import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import joblib
from utl import get_classes, parse_csv, get_accuracy

path_rgb_test = 'softmax_val/rgb/softmax_values.csv'
path_sgm_test = 'softmax_val/sgm/softmax_values.csv'
path_of_test = 'softmax_val/of/softmax_values.csv'
path_gt = 'softmax_val/test_evalu.txt'

gt_test = get_classes(path_gt)
_,rows_rgb_test = parse_csv(path_rgb_test)
_,rows_sgm_test = parse_csv(path_sgm_test)
_,rows_of_test = parse_csv(path_of_test)

test_data = []
for i in range(0,len(rows_rgb_test)):
    H = np.concatenate((rows_rgb_test[i],rows_sgm_test[i],rows_sgm_test[i]))
    test_data.append(H)

le = LabelEncoder()
test_labels = le.fit_transform(gt_test)

model_path = 'classif_m/modelSVM.npy'
model = joblib.load(model_path)


classif = 0
if classif == 3:
    dec = model.predict_proba(test_data)
else:
    dec = model.decision_function(test_data)


top1,top3,top5 = get_accuracy(dec,gt_test)

print('top1: '+str(top1*100))
print('top3: '+str(top3*100))
print('top5: '+str(top5*100))
