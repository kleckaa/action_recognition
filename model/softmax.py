import csv
import numpy as np
from utl import parse_csv, get_classes, top_accuracy_of_multuiple_softmax

path_gt = 'softmax_val/test_evalu.txt'
gt = get_classes(path_gt)

path_rgb = 'softmax_val/rgb/softmax_values.csv'
path_sgm = 'softmax_val/sgm/softmax_values.csv'
path_of = 'softmax_val/of/softmax_values.csv'

header,rows_rgb = parse_csv(path_rgb)
_,rows_sgm = parse_csv(path_sgm)
_,rows_of = parse_csv(path_of)




# Pro spojeni modelu A
# alpha * rgb + beta * sgm + gamma * of
alpha = 0.94
beta = 0.61
gamma = 0.39
mdl = 'A'
top1,top3,top5 = top_accuracy_of_multuiple_softmax(rows_rgb, rows_sgm, rows_of, gt, alpha, beta, gamma,mdl)

print('spojeni '+mdl)
print('top1: '+str(top1*100))
print('top3: '+str(top3*100))
print('top5: '+str(top5*100))
print('')

# Pro spojeni modelu B
# rgb * ((beta * sgm) + (gamma * of))
beta = 0.35
gamma = 0.86
mdl = 'B'
top1,top3,top5 = top_accuracy_of_multuiple_softmax(rows_rgb, rows_sgm, rows_of, gt, 1, beta, gamma,mdl)
print('spojeni '+mdl)
print('top1: '+str(top1*100))
print('top3: '+str(top3*100))
print('top5: '+str(top5*100))
