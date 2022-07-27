import copy
import csv
import logging
import math
import os
import queue
import shutil
import time
import tkinter as tk
from threading import *
from tkinter import (HORIZONTAL, Button, Entry, Frame, Label, StringVar,
                     filedialog)
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Progressbar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from bitarray import test
from matplotlib import pyplot as plt
from matplotlib import transforms
from PIL import Image, ImageTk
from sklearn import datasets
from sklearn.metrics import *
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import sugeno_integral

logger = logging.getLogger(__name__)
torch.manual_seed(8)
  
class LungCancer(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        app_width = 600
        app_height = 820

        
        x = (screen_width/2) - (app_width/2)
        y = (screen_height/2.2) - (app_height/2)
        # self.geometry("600x900");
        self.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
        
        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        container.configure(bg='black')
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, splitData, deepLearning, testImage):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.configure(bg='black')
  
# first window frame startpage
  
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        mainLabel = tk.Label(self,text='Lung Cancer Detection: Lim Edition', font=('Arial',25,'bold'),fg="white", bg="black",height=2).pack()
        
        img = ImageTk.PhotoImage(Image.open('lung.png'))
        label = tk.Label(self, image=img, bg='black')
        label.img = img 
        
        splitDataButton = tk.Button(self, text ="Split Data", command = lambda : controller.show_frame(splitData) ,borderwidth=0, width=50, height=2, fg="pink", bg="black", font=("Helvetica", 18,'bold'))
        deepLearningButton = tk.Button(self, text ="Deep Learning", command = lambda : controller.show_frame(deepLearning), borderwidth=0, width=50, height=2, fg="pink", bg="black", font=("Helvetica", 18,'bold'))
        testImageButton = tk.Button(self, text ="Cancer Detection", command = lambda : controller.show_frame(testImage), borderwidth=0, width=50, height=2, fg="pink", bg="black", font=("Helvetica", 18,'bold'))
        quitButton = tk.Button(self, text="Exit", command=self.quit,borderwidth=0, width=50, height=2, fg="pink", bg="black", font=("Helvetica", 18,'bold'))
        
        label.pack(anchor='center')
        splitDataButton.pack(anchor='center', pady=(100,0))
        deepLearningButton.pack(anchor='center')
        testImageButton.pack(anchor='center')
        quitButton.pack(anchor='center')
  
  
# second window frame page1
class splitData(tk.Frame):
     
    def __init__(self, parent, controller):
        
        def inputFilesDir():

            self.inputFile = filedialog.askdirectory(title="Choose the location of the image")
            fileInput = os.path.abspath(self.inputFile)
            ROOT_DIR = fileInput
            number_of_images = {}
            
            if os.path.exists("./data/"):
                    shutil.rmtree("./data/")
                
            for dir in os.listdir(ROOT_DIR):
                number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
            
            def dataFolder(p, split):
                    
                os.makedirs("./data/"+p, exist_ok=True)

                for dir in os.listdir(ROOT_DIR):
                    os.mkdir('./data/'+p+'/' + dir)

                    for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR, dir)), size = (math.floor(split*number_of_images[dir])-5), replace=False):
                        O = os.path.join(ROOT_DIR,dir,img)
                        D = os.path.join('./data/'+p,dir)
                        shutil.copy(O,D)
                            
                            
            dataFolder("train",0.8)
            dataFolder("val",0.2)
                
        def step():
                
            for i in range(5):
                self.update_idletasks()
                pb1['value'] += 30
                time.sleep(1)
            os.startfile(os.path.abspath('./data'))

                
        tk.Frame.__init__(self, parent)
        mainLabel = tk.Label(self,text='Split Data', font=('Arial',25,'bold'),fg="white", bg="black",height=2).pack()
        img = ImageTk.PhotoImage(Image.open('lung.png'))
        label = tk.Label(self, image=img, bg='black')
        label.img = img 
        label.pack()
        pb1 = Progressbar(self, orient=HORIZONTAL, length=500)
        pb1.pack(pady=(100,0))

        # b1 = tk.Button(self, text="Choose Folder",font=('Helvatica',16,'bold'), command=lambda:[threading.Thread(target=inputFilesDir).start(),threading.Thread(target=step).start()], height=2, width= 80, borderwidth=0, fg="pink", bg="black")
        b1 = tk.Button(self, text="Choose Folder",font=('Helvatica',16,'bold'), command=lambda:[inputFilesDir(),step()], height=2, width= 80, borderwidth=0, fg="pink", bg="black")
        b1.pack(pady=(20,0))
        b2 = tk.Button(self, text="Back To Main Page",font=('Helvatica',15,'bold'),command=lambda : controller.show_frame(StartPage),height=2, width= 80, borderwidth=0, fg="pink", bg="black")
        b2.pack()
        
class QueueHandler(logging.Handler):
    """Class to send logging records to a queue
    It can be used from different threads
    The ConsoleUi class polls this queue to display records in a ScrolledText widget
    """
    # Example from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06
    # (https://stackoverflow.com/questions/13318742/python-logging-to-tkinter-text-widget) is not thread safe!
    # See https://stackoverflow.com/questions/43909849/tkinter-python-crashes-on-new-thread-trying-to-log-on-main-thread
 
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
 
    def emit(self, record):
        self.log_queue.put(record)
  
# third window frame page2
class deepLearning(tk.Frame):
    
    def display(self, record):
            msg = self.queue_handler.format(record)
            self.scrolled_text.configure(state='normal')
            self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
            self.scrolled_text.configure(state='disabled')
 
            # Autoscroll to the bottom
            self.scrolled_text.yview(tk.END)
            
            
    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
            while True:
                try:
                    record = self.log_queue.get(block=False)
                except queue.Empty:
                    break
                else:
                    self.display(record)
            self.after(100, self.poll_log_queue)
            
    def __init__(self, parent, controller):
            
        def imshow(inp, title):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            plt.title(title)
            plt.show()

        def plot(val_loss,train_loss,typ):
            data_dir="./data/"
            plt.title("{} after epoch: {}".format(typ,len(train_loss)))
            plt.xlabel("Epoch")
            plt.ylabel(typ)
            plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train "+typ)
            plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation "+typ)
            plt.legend()
            plt.savefig(os.path.join(data_dir,typ+".png"))
        #     plt.figure()
            plt.close()

        def train_model(model, criterion, optimizer, scheduler,image_datasets,data_dir, num_epochs=25,model_name = "kaggle",):
            torch.manual_seed(8)
            data_dir="./data/"
            # edit epoch here
            num_epochs=1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            val_loss_gph=[]
            train_loss_gph=[]
            val_acc_gph=[]
            train_acc_gph=[]
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                logger.log(logging.INFO, 'Epoch {}/{}'.format(epoch+1, num_epochs))
                # print('Epoch {}/{}'.format(epoch+1, num_epochs)) 
                logger.log(logging.INFO, '-' * 10)
                # print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.        
                    for inputs, labels in dataloaders[phase]:
                        
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1) #was (outputs,1) for non-inception and (outputs.data,1) for inception
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds.detach() == labels.data)

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    if phase == 'train':
                        train_loss_gph.append(epoch_loss)
                        train_acc_gph.append(epoch_acc.item())
                    if phase == 'val':
                        val_loss_gph.append(epoch_loss)
                        val_acc_gph.append(epoch_acc.item())
                    
                    plot(val_loss_gph,train_loss_gph, "Loss")
                    plot(val_acc_gph,train_acc_gph, "Accuracy")

                    logger.log(logging.INFO, '{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                    # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    #     phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), data_dir+"/"+model_name+".pth")
                        logger.log(logging.INFO, '==>Model Saved')
                        # print('==>Model Saved')

                logger.log(logging.INFO, '')
                # print()
                

            time_elapsed = time.time() - since
            logger.log(logging.INFO, 'Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            # print('Training complete in {:.0f}m {:.0f}s'.format(
                # time_elapsed // 60, time_elapsed % 60))
            logger.log(logging.INFO, 'Best val Acc: {:4f}'.format(best_acc))
            # print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
            
            logger.log(logging.INFO, 'Getting the Probability Distribution')
            # print("\nGetting the Probability Distribution")
            testloader=torch.utils.data.DataLoader(image_datasets['val'],batch_size=1)
            model=model.eval()

            correct = 0
            total = 0
            f = open(data_dir+'/'+model_name+".csv",'w+',newline = '')
            writer = csv.writer(f)

            with torch.no_grad():
                num = 0
                temp_array = np.zeros((len(testloader),num_classes))
                for data in testloader:
                    images, labels = data
                    labels=labels.cuda()
                    outputs = model(images.cuda())
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cuda()).sum().item()
                    prob = torch.nn.functional.softmax(outputs, dim=1)
                    temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
                    num+=1
            # print("Accuracy = ",100*correct/total)
            logger.log(logging.INFO, "Accuracy = "+str(100*correct/total))
            # print("Accuracy = ",100*correct/total)

            for i in range(len(testloader)):
                writer.writerow(temp_array[i].tolist())
            f.close()
            return model

        # Getting Proba distribution
        # def get_probability(image_datasets,model,data_dir,model_name):
        #     data_dir="./data/"
        #     logger.log(logging.INFO, 'Getting the Probability Distribution')
        #     # print("\nGetting the Probability Distribution")
        #     testloader=torch.utils.data.DataLoader(image_datasets['val'],batch_size=1)
        #     model=model.eval()

        #     correct = 0
        #     total = 0
        #     import csv

        #     import numpy as np
        #     f = open(data_dir+'/'+model_name+".csv",'w+',newline = '')
        #     writer = csv.writer(f)

        #     with torch.no_grad():
        #         num = 0
        #         temp_array = np.zeros((len(testloader),num_classes))
        #         for data in testloader:
        #             images, labels = data
        #             labels=labels.cuda()
        #             outputs = model(images.cuda())
        #             _, predicted = torch.max(outputs, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels.cuda()).sum().item()
        #             prob = torch.nn.functional.softmax(outputs, dim=1)
        #             temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
        #             num+=1
        #     print("Accuracy = ",100*correct/total)
        #     logger.log(logging.INFO, "Accuracy = ",100*correct/total)
        #     # print("Accuracy = ",100*correct/total)

        #     for i in range(len(testloader)):
        #         writer.writerow(temp_array[i].tolist())
        #     f.close()
        
            
        def learningPart(self, data_dir,num_epochs):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # data_dir="./data/"
            
            global data_transforms
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
            }

            global image_datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
            
            global dataloaders
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                        shuffle=True, num_workers=4)
                        for x in ['train', 'val']}
            
            global dataset_sizes
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            
            global class_names
            class_names = image_datasets['train'].classes
            
            global num_classes
            num_classes = len(class_names)
            
            global device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print(class_names)
            logger.log(logging.INFO, class_names)

        #Get probability distributions from the 4 models
            # global num_epochs
            num_epochs = 1

            global criterion
            criterion = nn.CrossEntropyLoss()
            
            # Get a batch of training data
            global inputs, classes
            inputs, classes = next(iter(dataloaders['train']))

            model = models.vgg11_bn(pretrained = True)
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
            step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
            num_ftrs = model.classifier[0].in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model = model.to(device)
            logger.log(logging.INFO, '\nVGG-11')
            # model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_vgg11')
            model = Thread(target=train_model(model, criterion, optimizer, step_lr_scheduler,image_datasets,data_dir ,num_epochs=num_epochs, model_name = 'Kaggle_vgg11')).start()
            # get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11')
        
            
            model = models.googlenet(pretrained = True)
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
            step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            model = model.to(device)
            logger.log(logging.INFO, '\nGoogleNet')
            # model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_googlenet')
            model = Thread(target=train_model(model, criterion, optimizer, step_lr_scheduler,image_datasets,data_dir ,num_epochs=num_epochs, model_name = 'Kaggle_googlenet')).start()
            # get_probability(image_datasets,model,data_dir,model_name='Kaggle_googlenet')

            model = models.squeezenet1_1(pretrained = True)
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
            step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model = model.to(device)
            logger.log(logging.INFO, '\nSqueezeNet')
            # model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_squeezenet')
            model = Thread(target=train_model(model, criterion, optimizer, step_lr_scheduler,image_datasets,data_dir ,num_epochs=num_epochs, model_name = 'Kaggle_squeezenet')).start()
            # get_probability(image_datasets,model,data_dir,model_name='Kaggle_squeezenet')

            model = models.wide_resnet50_2(pretrained = True)
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
            step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            model = model.to(device)
            logger.log(logging.INFO, '\nWideResNet-50-2')
            # model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_wideresnet')
            model = Thread(target=train_model(model, criterion, optimizer, step_lr_scheduler,image_datasets,data_dir ,num_epochs=num_epochs, model_name = 'Kaggle_wideresnet')).start()
            # get_probability(image_datasets,model,data_dir,model_name='Kaggle_wideresnet')
            
            prob1,labels = getfile("Kaggle_vgg11",root = data_dir)
            prob2,_ = getfile("Kaggle_squeezenet",root = data_dir)
            prob3,_ = getfile("Kaggle_googlenet",root = data_dir)
            prob4,_ = getfile("Kaggle_wideresnet",root = data_dir)

            ensemble_sugeno(labels,prob1,prob2,prob3,prob4)
        
        def threading():
            t1=Thread(target=learningPart(self,"./data/",100))
            t1.start()
            
        def getfile(filename, root="../"):
            file = root+filename+'.csv'
            df = pd.read_csv(file,header=None)
            df = np.asarray(df)

            labels=[]
            for i in range(19):
                labels.append(0)
            for i in range(107):
                labels.append(1)
            for i in range(78):
                labels.append(2)
            labels = np.asarray(labels)
            return df,labels

        def predicting(ensemble_prob):
            prediction = np.zeros((ensemble_prob.shape[0],))
            for i in range(ensemble_prob.shape[0]):
                temp = ensemble_prob[i]
                t = np.where(temp == np.max(temp))[0][0]
                prediction[i] = t
            return prediction

        def metrics(labels,predictions,classes):
            # print("Classification Report:")
            logger.log(logging.INFO, 'Classification Report:')
            # print(classification_report(labels, predictions, target_names = classes,digits = 4))
            logger.log(logging.INFO, classification_report(labels, predictions, target_names = classes,digits = 4))
            matrix = confusion_matrix(labels, predictions)
            tn=matrix[0][0]
            tp=matrix[1][1]
            fp=matrix[0][1]
            fn=matrix[1][0]
            print(tn)
            print(tp)
            print(fn)
            print(fp)
            # print("Confusion matrix:")
            logger.log(logging.INFO, 'Classification matrix:')
            # print(matrix)
            logger.log(logging.INFO, matrix)
            # print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
            logger.log(logging.INFO, "\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
            # print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
            logger.log(logging.INFO, "\nBalanced Accuracy Score: "+str(100*balanced_accuracy_score(labels,predictions)))
            

        #Sugeno Integral
        def ensemble_sugeno(labels,prob1,prob2,prob3,prob4):
            prob1.shape[1]
            Y = np.zeros(prob1.shape,dtype=float)
            for samples in range(prob1.shape[0]):
                for classes in range(prob1.shape[1]):
                    X = np.array([prob1[samples][classes], prob2[samples][classes], prob3[samples][classes], prob4[samples][classes] ])
                    measure = np.array([1.5, 1.5, 0.01, 1.2])
                    X_agg = sugeno_integral.sugeno_fuzzy_integral_generalized(X,measure)
                    Y[samples][classes] = X_agg

            sugeno_pred = predicting(Y)

            correct = np.where(sugeno_pred == labels)[0].shape[0]
            total = labels.shape[0]
            logger.log(logging.INFO, "Accuracy =" + str(correct/total))
            # print("Accuracy = ",correct/total)
            classes = ['Benign','Malignant','Normal']
            metrics(sugeno_pred,labels,classes)
        
            
        tk.Frame.__init__(self, parent)
        
        mainLabel = tk.Label(self,text='Deep Learning', font=('Arial',25,'bold'),fg="white", bg="black",height=2).pack()
        b1 = tk.Button(self, text="Start Deep Learning", command= threading,borderwidth=0, width=50, height=3, fg="pink", bg="black", font=("Helvetica", 16,'bold'))
        b1.pack()
        
        self.scrolled_text=ScrolledText(self, state='disabled', height=35)
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.after(100, self.poll_log_queue)
        # T = tk.ScrolledText(self, height = 35, width = 70, state='disable')
        self.scrolled_text.pack()
        
        b2 = tk.Button(self, text="Back To Main Page",command=lambda : controller.show_frame(StartPage),borderwidth=0, width=50, height=3, fg="pink", bg="black", font=("Helvetica", 16,'bold'))
        b2.pack()

        
class testImage(tk.Frame):
    def __init__(self, parent, controller):
        
        # open a image file from hard-disk
        def open_image(initialdir='/'):
            global file_path
            file_path  = filedialog.askopenfilename(initialdir=initialdir, filetypes = [ ('Image File', '*.*' ) ]  )
            img_var.set(file_path)

            image = Image.open(file_path)
            image = image.resize((320,180)) # resize image to 32x32
            photo = ImageTk.PhotoImage(image)

            img_label = Label(middle_frame, image=photo, padx=10, pady=10)
            img_label.image = photo # keep a reference!
            img_label.grid(row=3, column=1)

            return file_path

        # #####################  Test Image
        def test_image():
            open_image()
            img = Image.open(file_path)
            mean = [0.485, 0.456, 0.406] 
            std = [0.229, 0.224, 0.225]
            transform_norm = transforms.Compose([transforms.ToTensor(), 
            transforms.Resize((224,224)),transforms.Normalize(mean, std)])
            # get normalized image
            img_normalized = transform_norm(img).float()
            img_normalized = img_normalized.unsqueeze_(0)
            # input = Variable(image_tensor)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_normalized = img_normalized.to(device)
            # print(img_normalized.shape)
            with torch.no_grad():
                model = torchvision.models.googlenet(pretrained=True)
                model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
                # print(model)
                model.load_state_dict(torch.load("extraFile\\Kaggle_googlenet.pth"))
                model.state_dict()
                model.eval().cuda()
                output =model(img_normalized)
                index = output.data.cpu().numpy().argmax()
                train_ds = ['Benign','Malignant','Normal']
                global class_name
                class_name = train_ds[index]
                test_result_var.set("Test Result: " + class_name)
                return class_name


        """  Top Frame  """
        # tl = Label(top_frame, text="Top frame").pack()
        # ##### H5 #################
        tk.Frame.__init__(self, parent)
        
        top_frame = Frame(self, bd=10,bg="black")
        top_frame.pack()

        middle_frame = Frame(self,bd=10,bg="black")
        middle_frame.pack()

        bottom_frame = Frame(self, bd=10,bg="black")
        bottom_frame.pack()


        notification_frame = Frame(self,bd=10,bg="black")
        notification_frame.pack()

        mainLabel = tk.Label(top_frame,text='Cancer Detection', font=('Arial',25,'bold'),fg="white", bg="black",height=2).grid(row=1, column=1, columnspan=2)

        #######   IMAGE input
        btn_test = Button(top_frame, text='Press to Choose Image',  command = test_image , font=('Helvetica',16,'bold'), height=2, borderwidth=0, fg="pink", bg="black")
        btn_test.grid(row=2, column=1,columnspan=2, pady=(80,0))
        dir_input = Label(top_frame,font=("Helvetica", 15,'bold'),bg="black", fg="white", text="File Directory: ").grid(row=7, column=1)
        img_var = StringVar()
        img_var.set("/")
        img_entry = Entry(top_frame, textvariable=img_var, width=60)
        
        img_entry.grid(row=7, column=2)

        """ middle Frame  """
        ml = Label(middle_frame, font=("Courier", 10),bg="black", fg="white", text="Image Shown Below").grid(row=1, column=1)


        """ bottom Frame  """
        
        # Test Image butttom
        


        test_result_var = StringVar()
        test_result_var.set("Test Result:")
        test_result_label = Label(bottom_frame, textvariable=test_result_var, font=('Helvetica',20,'bold'), height=2, width= 80, borderwidth=0, fg="white", bg="black").pack()
        b2 = tk.Button(self, text="Back To Main Page",font=('Helvetica',16,'bold'),command=lambda : controller.show_frame(StartPage),height=2, width= 80, borderwidth=0, fg="pink", bg="black")
        b2.pack() 
  
  
# Driver Code
if __name__ == '__main__':  
    logging.basicConfig(level=logging.DEBUG)  
    app = LungCancer()
    app.title('Lim LCD')
    app.iconbitmap('lung_icon.ico')
    app.mainloop()
