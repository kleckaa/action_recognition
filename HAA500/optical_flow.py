import os
import cv2
from extract_tool import get_dict
from torchvision.io import read_video
from seg_video_paser import segm_img
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
from torchvision import transforms
from torchvision.io import write_jpeg
from PIL import Image

if torch.cuda.is_available():
    device = 'cuda:0'
else :
    device = 'cpu'

model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_K_V2, progress=False).to(device) #Nacteni modelu
model = model.eval()

weights = Raft_Large_Weights.C_T_SKHT_K_V2
transforms = weights.transforms()

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960])
    #img1_batch = F.resize(img1_batch, size=[544, 960])
    img2_batch = F.resize(img2_batch, size=[520, 960])
    #img2_batch = F.resize(img2_batch, size=[544, 960])
    return transforms(img1_batch, img2_batch)

path = 'video/' # Cesta pro videa z HAA500
classes, data = get_dict(path)

data_path = 'opt_flow/' # Cesta pro vystup part1
exit_file = 'videos/' # Cesta pro vystup part2

new_path = 'opt_flow/videos'

for i in range(0,len(classes)):
    for j in range(0,len(data[classes[i]])):

        video_path = 'video/' + classes[i] + '/' + data[classes[i]][j]
        frames, _, _ = read_video(str(video_path), pts_unit='sec')
        frames = frames.permute(0, 3, 1, 2)

        newpath = data_path + exit_file + data[classes[i]][j].replace('.mp4', '')

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        for frame in range(0,len(frames)-1):

            img1_batch = torch.stack([frames[frame]])
            img2_batch = torch.stack([frames[frame+1]])

            img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

            img1_batch = img1_batch.float()
            img2_batch = img2_batch.float()

            #print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            #print(f"type = {type(list_of_flows)}")
            #print(f"length = {len(list_of_flows)} = number of iterations of the model")

            predicted_flows = list_of_flows[-1]
            #print(f"dtype = {predicted_flows.dtype}")
            #print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
            #print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

            flow_img = flow_to_image(predicted_flows)
            flow_img = flow_img.permute(0, 2, 3, 1)
            #img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            #write_jpeg(flow_img, 'opt_flow' + f"predicted_flow_{1}.jpg")

            r = flow_img.cpu().numpy()

            del flow_img
            del predicted_flows
            del list_of_flows
            del img1_batch
            del img2_batch

            pix = r[0,:,:,:]

            #rr = r.convert("RGB")
            #pix = np.array(rr)

            pix = cv2.resize(pix, (224, 224))

            #newpath = 'opt_flow'
            #frame = 0
            cv2.imwrite(newpath + '/' + 'image%03d.jpg' % frame, pix)


        if j == len(data[classes[i]]) - 4:
            with open(data_path+'val.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame) + ' ' + str(i))
                f.write('\n')

            with open(data_path+'val2.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1) + ' ' + str(i))
                f.write('\n')

        elif j > len(data[classes[i]])-4:
            with open(data_path+'test.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame))
                f.write('\n')

            with open(data_path+'test2.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1))
                f.write('\n')

            with open(data_path+'test_evalu.txt','a') as f:
                f.write(str(i))
                f.write('\n')

        else:
            with open(data_path+'train.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame) + ' ' + str(i))
                f.write('\n')

            with open(data_path+'train2.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1) + ' ' + str(i))
                f.write('\n')

print('done')