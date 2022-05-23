from glob import glob
from os import walk


#path = 'video/'

def get_dict(path):

    data = dict()
    classes = list()

    classes_ex = list(glob(path + '*', recursive=True))
    for i in range(0,len(classes_ex)):
        classes_ex[i]=classes_ex[i].replace('video\\', '')

        f = []
        for (dirpath, dirnames, filenames) in walk(path + '/' + classes_ex[i]):
            f.extend(filenames)
            break

        data[classes_ex[i]] = f
        classes.append(classes_ex[i])


    return classes,data










