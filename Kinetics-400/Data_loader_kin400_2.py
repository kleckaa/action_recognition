import json
from pytube import YouTube, Playlist
from moviepy.editor import *
import pickle
#import time
#import os
import datetime

from pytube.exceptions import VideoUnavailable, VideoPrivate, MaxRetriesExceeded
from urllib.error import URLError



def download_kin(path_dataset, target_path):

    f = open(path_dataset)
    data = json.load(f)
    link_first = 'http://youtube.com/watch?v='
    data_list = dict()



    for i in range(0, 100):
        ind = list(data.keys())[i]
        yt_link = data[ind]['url']
        duration = int(data[ind]['duration'])
        start = int(data[ind]['annotations']['segment'][0])
        end = int(data[ind]['annotations']['segment'][1])
        label = data[ind]['annotations']['label']
        name_vid = 'vid' + str(i)

        print(yt_link + '?t=' + str(start))
        start_time = str(datetime.timedelta(seconds=start))
        end_time = str(datetime.timedelta(seconds=end))

        try:
             yt = YouTube(yt_link)
             yt.check_availability()
        except VideoUnavailable:
            pass
        except VideoPrivate:
            pass
        else:
            try:
                video = yt.streams.filter(res='240p')
                video.fmt_streams[0].download(output_path=target_path, filename='vid_main' + '.mp4')
            except MaxRetriesExceeded:
                pass
            except URLError:
                pass
            except KeyError:
                pass
            else:

                vid_load = VideoFileClip(target_path + '/vid_main' + '.mp4').subclip(start, end)

                vid_load.write_videofile(target_path+ name_vid + '.mp4')

                dl = {'file': name_vid + '.mp4', 'label': label}

                data_list[name_vid + '.mp4'] = dl
                data_list['name'] = name_vid+'.mp4'
                data_list['name']['label'] = label

    pickle.dump(data_list, open('data_l.p', 'wb'))



path_dataset = 'kinetics400/train.json'
target_path = 'videos/vid'

download_kin(path_dataset, target_path)
print('end')