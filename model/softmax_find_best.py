from utl import parse_csv, get_classes, top_accuracy_of_multuiple_softmax

path_gt = 'data/gt_val.txt'
gt = get_classes(path_gt)

path_rgb = 'data/rgb/softmax_values.csv'
path_sgm = 'data/sgm/softmax_values.csv'
path_of = 'data/of/softmax_values.csv'

header,rows_rgb = parse_csv(path_rgb)
_,rows_sgm = parse_csv(path_sgm)
_,rows_of = parse_csv(path_of)


mdl = 'A'
alpha = 0
beta = 0
gamma = 0
inc = 0.01

top1_list = list()
other_info = list()

for i in range(0,100):
    alpha = 1
    for j in range(0,100):
        for k in range(0,100):

            top1,top3,top5 = top_accuracy_of_multuiple_softmax(rows_rgb, rows_sgm, rows_of, gt, alpha, beta, gamma,mdl)
            print('alpha: '+str(alpha))
            print('beta: ' + str(beta))
            print('gamma: ' + str(gamma))
            print('top1: ' + str(top1 * 100))
            print('top3: ' + str(top3 * 100))
            print('top5: ' + str(top5 * 100))
            print('---')

            top1_list.append(top1)
            other_info.append('top1: ' + str(top1 * 100)+', top3: ' + str(top3 * 100)+  ', top5: ' + str(top5 * 100) + ', alpha: '+str(alpha)+', beta: ' + str(beta)+', gamma: ' + str(gamma))

            gamma += inc

        beta += inc
        gamma = 0

    alpha += inc
    beta = 0

#for j in range(0,200):
#
#    tmp = max(top1_list)
#    index = top1_list.index(tmp)
#
#    print(str(j) + '.: '+other_info[index])
#
#    top1_list[index] = 0

for j in range(0,2000):

    tmp = max(top1_list)
    index = top1_list.index(tmp)

    print(str(j) + '.: '+other_info[index])
    with open('data/vysl_krat_krat.txt', 'a') as f:
        f.write(str(j) + ': '+other_info[index] )
        f.write('\n')

    top1_list[index] = 0