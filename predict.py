import sys
import time
import json
import torch
import argparse
import numpy as np
from PIL import Image
from torch import nn , optim
import matplotlib.pyplot as plt 
from torchvision import datasets , models , transforms


def load_model():
    print("load your model")
    
    checkpoint = torch.load(args.model_checkpoint)
    model = checkpoint.get('model')
    model.classifier = checkpoint.get("classifier")
    model.load_state_dict(checkpoint.get('state_dict')) 
    
    return model
  
def process_image(image , size=(256, 256), crop_size=244):
    print("preprocessing image")
        
    #Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    image = image.resize(size, Image.ANTIALIAS)
    hight, width  = size
    dim = {
        "left": (width - crop_size) / 2,
        "lower": (hight - crop_size) / 2,
        "right": ((width - crop_size) / 2) + crop_size,
        "top": ((hight - crop_size) / 2) + crop_size,
    }
  
    cropped_image = image.crop(tuple(dim.values()))    
    
    image_array = np.array(cropped_image)/ (size[0] - 1)
    image_array = (image_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return image_array
  
    
def classify(image_path, topk=5):
    
    topk=int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image = image.unsqueeze_(0)
        image = image.float()    
        
        model = load_model()
        
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        
        model.eval()
        log = model.forward(image.permute(0, 3, 1, 2))    
        prop = torch.exp(log)
        top_p, top_class = prop.topk(topk)
        
        return top_p[0], top_class[0]
    
    
def find_categories():
    
    if (args.category_names is not None):
        cat_file = args.category_names 
        file = json.loads(open(cat_file).read())
        return file
        
        
def display(probs,classes):
    print("display prediction")
    cat_file = find_categories()
    
    classes = classes.cpu().detach().numpy()
    plant_classes = [cat_file[str(cls)] for cls in classes]
    im = Image.open(image_path)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im);
    y_positions = np.arange(len(plant_classes))
    ax[1].barh(y_positions,probs,color='blue')
    ax[1].set_yticks(y_positions)
    ax[1].set_yticklabels(plant_classes)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Accuracy (%)')
    ax[0].set_title('Top 5 Flower Predictions')
    
    return None

        
        
def parse():
    parser = argparse.ArgumentParser(description='use a trained neural network to predict an image label!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args  


def main():
    global args
    args = parse()     
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    
    
    image_path = args.image_input
    
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
        
    probs,classes = classify(image_path,top_k)
    
    display(probs,classes)
    
    return

main()    