import json

import time
import os

def download_kin(path_dataset, target_path):

    f = open(path_dataset)
    data = json.load(f)
    link_first = 'http://youtube.com/watch?v='
    data_list = dict()

    start = time.perf_counter()

    for i in range(0,10000):
        ind = list(data.keys())[i]
        yt_link = data[ind]['url']
        duration = int(data[ind]['duration'])
        start = int(data[ind]['annotations']['segment'][0])
        end = int(data[ind]['annotations']['segment'][1])
        label = data[ind]['annotations']['label']
        name_vid = 'vid' + str(i)

        #print(yt_link+'?t='+str(start))
        #start_time = str(datetime.timedelta(seconds=start))
        #end_time = str(datetime.timedelta(seconds=end))


        filename = target_path + 'kinet_'+str(i)+'.mp4'

        os.system('youtube-dl --external-downloader ffmpeg --external-downloader-args ' + '"-ss '+ start_time + ' -to ' + end_time +'" '+ '-f 18 "'+yt_link+'" ' + '-o ' + '"'+filename+'"' )
        #os.system('youtube-dl -f 18 --postprocessor-args ' + '"-ss ' + start_time + ' -to ' + end_time + '" ' + '"' + yt_link + '"')



path_dataset = 'kinetics400/train.json'
target_path = 'videos/'

download_kin(path_dataset, target_path)
print('end')