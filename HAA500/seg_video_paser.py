import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

def segm_img(model, input_image, colors):

    #https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ # Odkaď

    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    #model.eval()


    input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        #print('cuda:0')
        input_batch = input_batch.to('cuda:0')
        model.to('cuda:0')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)


    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)


    rr = r.convert("RGB")
    pix = np.array(rr)

    return pix

