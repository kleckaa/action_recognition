import csv
import numpy as np

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

def get_classes(path):
    f = open(path, 'r')
    lines_ground = list()
    for x in f:
        lines_ground.append(int(x.replace('\n', '')))

    return lines_ground


def top_accuracy_of_multuiple_softmax(rows_rgb, rows_sgm, rows_of, gt, alpha, beta, gamma, mdl):
    top1_good = 0
    top3_good = 0
    top5_good = 0
    for i in range(0, len(rows_rgb)):
    #for i in range(0,1):

        if mdl == 'A':
            new_softmax = (alpha * rows_rgb[i]) + (beta * rows_sgm[i]) + (gamma * rows_of[i])
        elif mdl == 'B':
            new_softmax = (rows_rgb[i]) * ((beta * rows_sgm[i]) + (gamma * rows_of[i]))
        else:
            print('err')

        ind = np.argpartition(new_softmax, -5)[-5:]
        values = new_softmax[ind]
        x = np.argsort(values)
        ind = ind[x]
        ind = np.flip(ind)
        #print(ind)
        #print('')

        #print(new_softmax[378])
        #print(new_softmax[381])
        #print(new_softmax[0])

        with open('data/t_plus_plus.csv', 'a') as f:
            f.write('-1;'+ str(ind[0]) + ';' + str(ind[1]) + ';' + str(ind[2]) + ';' + str(ind[3]) + ';' + str(ind[4]) + ';')
            f.write('\n')

        top3_classes = [ind[0], ind[1], ind[2]]

        if gt[i] == ind[0]:
            top1_good += 1
        if gt[i] in top3_classes:
            top3_good += 1
        if gt[i] in ind:
            top5_good += 1

    return top1_good / (i + 1), top3_good / (i + 1), top5_good / (i + 1)

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