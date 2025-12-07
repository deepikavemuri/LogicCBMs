import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn.functional as F
import json
import torch.optim as optim
from tqdm import tqdm
from difflogic import LogicLayer

class CLEVRLoader(nn.Module):
    def __init__(self, dataset_path='./data/logically_clevr/images_reduced/'):
        super(CLEVRLoader, self).__init__()
        transform = transforms.Compose([
                transforms.Resize((224, 224)),        
                transforms.ToTensor(),                
                transforms.Normalize(                 
                    mean=[0.485, 0.456, 0.406],       
                    std=[0.229, 0.224, 0.225] 
                )
            ])
        self.data = datasets.ImageFolder(root=dataset_path, transform=transform)
        with open("./data/logically_clevr/scene_logic1.json", "r") as file:
            self.class0_json = json.load(file)
        with open("./data/logically_clevr/scene_logic2.json", "r") as file:
            self.class1_json = json.load(file)
        with open("./data/logically_clevr/scene_logic3.json", "r") as file:
            self.class2_json = json.load(file)

    def forward(self, x):
        return x 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, _ = self.data.samples[idx]
        file_name = img_path.split('/')[-1]
        img, label = self.data[idx]
        
        if label == 0:
            self.scene_json = self.class0_json
        elif label == 1:
            self.scene_json = self.class1_json
        elif label == 2:
            self.scene_json = self.class2_json

        for scene in self.scene_json["scenes"]:
            if scene["image_filename"] == file_name:
                concepts = scene["objects"][0]["concepts"].values()
        return img, label, list(concepts)

class CBM(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super(CBM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.concept_layer = nn.Linear(512, num_concepts)
        self.logic_classifier = nn.Sequential(
            LogicLayer(4, 25, connections='full', fixed_gates=True),
            LogicLayer(25, 20, connections='full', fixed_gates=True),
            nn.Linear(20, 3)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        concepts = torch.sigmoid(self.concept_layer(x))
        classes = torch.softmax(self.logic_classifier(concepts), dim=1)
        return concepts, classes
    

def train(args, trainloader, model):
    concept_loss = nn.BCELoss()
    class_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(args.epochs):
        model.train()
        print(f"Epoch {epoch}/{args.epochs}")
        running_loss = 0
        attr_acc = 0
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{args.epochs}') as pbar:
            for i, batch in enumerate(trainloader):
                x, y, c = batch
                x = x.cuda()
                y = y.cuda()
                c = torch.stack(c, dim=1).float().cuda()
            
                optimizer.zero_grad()
                concepts_pred, classes = model(x)

                c_loss = concept_loss(concepts_pred.squeeze(0), c.squeeze(0))
                y_loss = class_loss(classes, y)
                loss = c_loss + y_loss
                running_loss += loss.item()

                attr_acc += (torch.round(concepts_pred) == c).float().mean()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item()) 
                pbar.update(1)
        
        concept_acc = attr_acc.item()/len(trainloader)
        print(f"Loss: {running_loss/len(trainloader)}, Concept Accuracy: {concept_acc}")
        if concept_acc == 1.0:
            break

    torch.save(model, args.save_dir + 'clevr_logic_model.pth')       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./saved_models/', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=int, default=1e-5, help='learning rate')
    parser.add_argument('--num_concepts', type=int, default=4, help='Number of concepts')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')

    args = parser.parse_args()

    clevr_logic = CLEVRLoader()
    dataloader = DataLoader(clevr_logic, batch_size=2, shuffle=True, drop_last=True)
    model = CBM(args.num_concepts, args.num_classes).cuda()
    train(args, dataloader, model)