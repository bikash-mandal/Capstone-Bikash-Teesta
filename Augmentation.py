''' Module for data augmentation.
http://pytorch.org/docs/master/torchvision/transforms.html '''

#Created by: Biswas T.

from torchvision import transforms
import T_transforms
import pdb

class Augmentation:   
    def __init__(self,strategy):
        print ("Data Augmentation Initialized with strategy %s"%(strategy));
        self.strategy = strategy;
        
        
    def applyTransforms(self):
        if self.strategy == "H_FLIP": # horizontal flip with a probability of 0.5
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize([64,64]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([64,64]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        elif self.strategy == "SCALE_H_FLIP": # resize to 224*224 and then do a random horizontal flip.
            data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                #transforms.RandomHorizontalFlip(),
                T_transforms.NP_RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #T_transforms.Normalize([1.77213186, 1.79354817, 1.81675477, 1.82784351, 1.83484719, 1.84585792, 1.86831774],
                 #                    [1.87948111, 1.86608827, 1.84880492, 1.834254, 1.83398459, 1.83418879, 1.83613726])
            ]),
            'val': transforms.Compose([
                transforms.Scale([224,224]),
                T_transforms.NP_RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #T_transforms.Normalize([1.77213186, 1.79354817, 1.81675477, 1.82784351, 1.83484719, 1.84585792, 1.86831774],
                  #                   [1.87948111, 1.86608827, 1.84880492, 1.834254, 1.83398459, 1.83418879, 1.83613726])
            ]),
        }
        elif self.strategy == "TEST":
            data_transforms = {
            'train': transforms.Compose([    
                T_transforms.NP_RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                #T_transforms.NP_RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
        }
        else :
            print ("Please specify correct augmentation strategy : %s not defined"%(self.strategy));
            exit();
            
        return data_transforms;

