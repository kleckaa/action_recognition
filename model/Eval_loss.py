import numpy as np
import matplotlib.pyplot as plt

#path_train_loss = 'small_32f/1/model_t.txt'
#exit_path = 'small_32f/1'

def load_data(path_train_loss):
    f = open(path_train_loss, 'r')
    lines = list()
    for x in f:
        lines.append(x)

    train = list()
    val = list()
    traintop = list()
    valtop = list()
    for i in range(0, len(lines)):
        sup = lines[i].split('\t')
        kup = sup[0].split(' ')

        top1 = sup[2].split(' ')

        value = sup[1].split(' ')

        if kup[0].replace(':', '') == 'Train':
            train.append(float(value[1]))
            traintop.append(float(top1[1]))

        elif kup[0].replace(':', '') == 'Val':
            val.append(float(value[1]))
            valtop.append(float(top1[1]))
        else:
            print('Err')
            
    max_val_top = 0
    max_index = 0
    lowest_loss = 6
    lowest_index = 0
    for j in range(0, len(valtop)):
        if valtop[j] > max_val_top:
            max_val_top = valtop[j]
            max_index = j
        if val[j] < lowest_loss:
            lowest_loss = val[j]
            lowest_index = j

    #print(val)

    lowest_index +=1
    max_index += 1 
    
    #max_val_top = max(valtop)
    #max_index = valtop.index(max_val_top) + 1

    return train, val, traintop, valtop, max_index, lowest_index

def plot_graph_top(path_train_loss, exit_path, max_y, min_y, draw_vertical=False, depend_on_loss=False):
    train, val, traintop, valtop, max_index, lowest_index = load_data(path_train_loss)

    print('valtop: '+str(valtop[max_index-1]))
    print(max_index)
    print('lowestloss: '+str(valtop[lowest_index-1]))
    print(lowest_index)
    #print(max_index)    
    t = np.linspace(1, len(val), len(val))
    fig, ax = plt.subplots()
    ax.plot(t, traintop, label='Train')
    ax.plot(t, valtop, label='Val')
    if draw_vertical:
        if depend_on_loss:
            plt.axvline(x=lowest_index, label='Model', color='r')
        else:
            plt.axvline(x=max_index, label='Model', color='r')
    ax.legend()
    ax.set(xlabel='Epocha', ylabel='Top1 [%]',
           title='Průběh Top1 hodnoty při trénování')

    plt.ylim(min_y,max_y)



    fig.savefig(exit_path + '/top1.eps')
    plt.show()

def plot_graph_loss(path_train_loss, exit_path, max_y, min_y, draw_vertical=False, depend_on_loss=False ):

    train, val, traintop, valtop, max_index,lowest_index = load_data(path_train_loss)

    t = np.linspace(1, len(val), len(val))


    fig, ax = plt.subplots()
    ax.plot(t, train, label='Train')
    ax.plot(t, val, label='Val')
    if draw_vertical:
        if depend_on_loss:
            plt.axvline(x=lowest_index, label='Model', color='r')
        else:
            plt.axvline(x=max_index, label='Model', color='r')
    ax.legend()
    ax.set(xlabel='Epocha', ylabel='Loss',
           title='Průběh loss funkce')

    plt.ylim(min_y,max_y)



    fig.savefig(exit_path + '/loss.eps')
    plt.show()


