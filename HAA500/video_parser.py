import os
import cv2
from extract_tool import get_dict
from seg_video_paser import segm_img
import torch


#https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ # Odkaď
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()
#Segmentation = True #True jestli chceme segmentovane videa
Segmentation = False #False jestli chceme RGB videa

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

path = 'video/' # Cesta pro videa z HAA500
classes,data = get_dict(path)


data_path = 'data/'  #Kam part1
exit_file = 'videos/' #Kam part2

with open(data_path + 'classes.txt', 'w') as f:
    for i in range(0,len(classes)):
        f.write(str(i) + ' ' + classes[i])
        f.write('\n')

for i in range(0,len(classes)):
    for j in range(0, len(data[classes[i]])):
    #for j in range(0, 5):
        filename = path + classes[i] + '/' + data[classes[i]][j]

        newpath = data_path + exit_file + data[classes[i]][j].replace('.mp4', '')
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        vidcap = cv2.VideoCapture(filename) #Nacteni videa

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) #zjisteni delky videa

        success,image = vidcap.read()

        frame = 0

        while success:


            if Segmentation:
                image = segm_img(model, image, colors)

            image = cv2.resize(image, (224, 224))  # 224 x 224

            cv2.imwrite(newpath + '/' + 'image%03d.jpg' % frame, image)  # save frame as JPEG file
            frame += 1
            success, image = vidcap.read()

        if j == len(data[classes[i]]) - 4:
            with open(data_path+'val.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1) + ' ' + str(i))
                f.write('\n')

        elif j > len(data[classes[i]])-4:
            with open(data_path+'test.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1))
                f.write('\n')

            with open(data_path+'test_evalu.txt','a') as f:
                f.write(str(i))
                f.write('\n')

        else:
            with open(data_path+'train.txt', 'a') as f:
                f.write(newpath + ' 0 ' + str(frame-1) + ' ' + str(i))
                f.write('\n')



print('done')
    

