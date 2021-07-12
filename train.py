import os
import time
import json
import torch
import argparse
import numpy as np
from PIL import Image
from torch import nn , optim
import matplotlib.pyplot as plt 
from collections import OrderedDict
from workspace_utils import active_session
from torchvision import datasets , models , transforms

def valid():
    print("validating parameters")
    if(args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled.but no GPU detected")
        
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    
    data_dir = os.listdir(args.data_directory)
    
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('missing sub-directories')

    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet')      
    
def process_data(data_dir):
    print("processing data")
    train_dir, test_dir, valid_dir = data_dir 
    
    img_size = 224
    orig_img_size = 256
    mean = [0.485, 0.456, 0.406]
    stdDev = [0.229, 0.224, 0.225]

    data_transforms_train = transforms.Compose([

        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(img_size),# Randomly resized and cropped images to 224x224 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),# Convert to a tensor
        transforms.Normalize(mean,stdDev)
    ])


    data_transforms_valid = transforms.Compose([

        transforms.Resize(orig_img_size),
        transforms.CenterCrop(img_size), #cropped images to 224x224 
        transforms.ToTensor(),# Convert to a tensor
        transforms.Normalize(mean,stdDev)
    ])


    data_transforms_test = transforms.Compose([

        transforms.Resize(orig_img_size),
        transforms.CenterCrop(img_size),#cropped images to 224x224 
        transforms.ToTensor(),# Convert to a tensor
        transforms.Normalize(mean,stdDev)
    ])
    #Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir , transform = data_transforms_train)

    valid_dataset = datasets.ImageFolder(valid_dir , transform = data_transforms_valid )

    test_dataset = datasets.ImageFolder(test_dir , transform = data_transforms_test )


    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=34,shuffle = True)

    validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=34,shuffle = True)

    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=34,shuffle = True)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
            
    loaders = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    
    return loaders


def get_data():
    print("get data")
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'   
    test_dir = args.data_directory + '/test'
    data_dir = [train_dir,valid_dir,test_dir]
    
    return process_data(data_dir)


def classifier(model,input_node,hidden_units):
    
    #freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #create new classifier
    new_classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(input_node,hidden_units)),
        ('relu', nn.ReLU()),
        ('drop_out',nn.Dropout(p = 0.4)),
        ('fc2',nn.Linear(hidden_units,102)),
        ('output', nn.LogSoftmax(dim=1))        
    ]))
    
    return new_classifier

def build_model():
    print("build your model")
    
    if (args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch   
        
    if (arch_type == 'vgg'):
        model = models.vgg13(pretrained=True)
        input_node= 25088
        out_units = 4096
        
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node= 1024  
        out_units = 1000
        
    if (args.hidden_units is None):
        hidden_units = out_units
    else:
        hidden_units = args.hidden_units  
    
    for param in model.parameters():
        param.requires_grad = False
        
    #new classifier
    model.classifier = classifier(model,input_node,hidden_units)
    
    return model


#Do validation on the test set

def test(model,testloader,device='cpu'):
    print("test model")
    accuracy = 0
    model.eval()
    
    for images , labels in testloader:
        
        with torch.no_grad():
            
            log = model.forward(images)
            prop = torch.exp(log)
            top_p, top_class = prop.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item() 
            
    model_accuracy =  (accuracy / len(testloader)) * 100 
    return "Model accuracy is {} %".format(model_accuracy)


#define validation function
def validation(model, validloader, criterion):
    
    valid_loss = 0
    accuracy = 0
    
    for images , labels in validloader:
        
        #validation loss
        log = model.forward(images)
        valid_loss += criterion(log,labels).item()
        
        #accuracy
        prop = torch.exp(log)
        top_p, top_class = prop.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item() 
        
    return valid_loss , accuracy

def train(model,data,print_every = 20):
    print("train model")
    
    if (args.learning_rate is None):
        learning_rate = 0.001
    else:
        learning_rate = args.learning_rate
    
    if (args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.classifier.parameters(),lr=learning_rate)
    
    step = 0
    running_loss = 0
    
    with active_session():
        
        for epoch in range(epochs):
            for images , labels in  trainloader:
                #Move input and label tensors to the default device
                step +=1

                optimizer.zero_grad()

                log = model.forward(images)
                loss = criterion(log , labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if step % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        #validation function
                        validation_loss, validation_accuracy = validation(
                        model, validloader, criterion)

                        print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"validation loss: {validation_loss/len(validloader):.3f}.. "
                        f"validation accuracy: {validation_accuracy/len(validloader):.3f}")

                        running_loss = 0
                        model.train()
                        
    
    print("DONE !")
    test_result = test_accuracy(model,testloader,device)
    print('final accuracy on test set: {}'.format(test_result))    
    
def save_model(model):
    print("save your model")
    
    if (args.save_dir is None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir) 
    
    if not Path("checkpoint.pth").exists():
        save_checkpoint(model) 
    
def create_model():
    
    valid()
    data = get_data()
    model = build_model()
    model = train(model,data)
    
    save_model(model)
    
    return
    
    
        
def parse():
    print("add our argument")
    
    parser = argparse.ArgumentParser(description='Train a neural network!')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use and you have OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--gpu',action='store_true', help='to use gpu')
    args = parser.parse_args()
    
    return args
        
def main():
    print("creating a deep learning model") 
    
    global args
    args = parse()
    create_model()
    print("finished!")
    
    return

main()        