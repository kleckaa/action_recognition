def get_info_lol(path):
    f = open(path, 'r')
    lines_t = list()
    for x in f:
        lines = x.replace('\n', '')
        lines_t.append(lines.split(' '))

    return lines_t

def get_nvm(path_test_file, path_ground_truth):
    f = open(path_test_file, 'r')
    lines_test = list()
    for x in f:
        lines_test.append(x.replace('\n','').split(';'))

    f = open(path_ground_truth, 'r')
    lines_ground = list()
    for x in f:
        lines_ground.append(x.replace('\n',''))

    return lines_test,lines_ground

def calculate_info(K,lines_test,lines_ground,exit_path):

    class_info = [0] * len(K)
    action_info = [0] * 4

    for i in range(0, len(lines_test)):
        #print(lines_test[i][1])
        #print(lines_ground[i])
        if lines_test[i][1] == lines_ground[i]:
            class_info[int(lines_ground[i])] += 1

            action_info[int(K[int(lines_ground[i])][1])] += 1


    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(0,len(K)):
        if int(K[i][1]) == 0:
            i0 +=1
        elif int(K[i][1]) == 1:
            i1 += 1
        elif int(K[i][1]) == 2:
            i2 += 1
        elif int(K[i][1]) == 3:
            i3 += 1
        else:
            print('Ërr')

    kval = len(lines_test)/500

    i0 = i0 * kval
    i1 = i1 * kval
    i2 = i2 * kval
    i3 = i3 * kval

    ii = [i0,i1,i2,i3]

    with open(exit_path, 'a') as f:
        for i in range(0,len(action_info)):
            f.write('Akce '+str(i)+ ': '+ str(action_info[i]/ii[i]))
            f.write('\n')

        f.write('...')
        f.write('\n')
        f.write('\n')

        for i in range(0,len(K)):
            f.write(K[i][2] + ': '+str(class_info[i]/(kval)))
            f.write('\n')




path_rgb = 'softmax_val/rgb/softmax_values.csv'
path_sgm = 'softmax_val/sgm/softmax_values.csv'
path_of = 'softmax_val/of/softmax_values.csv'
gt_path = 'softmax_val/test_evalu.txt'
K = get_info_lol('analysis/classes.txt')


lines_rgb,lines_ground = get_nvm(path_rgb, gt_path )
lines_sgm,_ = get_nvm(path_sgm, gt_path )
lines_of,_ = get_nvm(path_of, gt_path )


exit_path = 'analysis/rgb.txt'
calculate_info(K,lines_rgb,lines_ground,exit_path)

exit_path = 'analysis/sgm.txt'
calculate_info(K,lines_sgm,lines_ground,exit_path)

exit_path = 'analysis/of.txt'
calculate_info(K,lines_of,lines_ground,exit_path)


'''
K = get_info_lol('analysis/classes.txt')

path_plus_plus = 'data/t_krat_plus.csv'
path_krat_plus = 'data/t_plus_plus.csv'


lines_plus_plus, lines_ground = get_nvm(path_plus_plus, gt_path )
lines_krat_plus, _ = get_nvm(path_krat_plus, gt_path )

exit_path = 'analysis/plus_plus.txt'
calculate_info(K,lines_plus_plus,lines_ground,exit_path)

exit_path = 'analysis/krat_plus.txt'
calculate_info(K,lines_krat_plus,lines_ground,exit_path)
'''

