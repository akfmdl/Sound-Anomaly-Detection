from torch.utils.data import DataLoader
import torchvision.models as models
from train.batch_dataset import BatchDataset
from train.split_dataset import split_dataset
import tqdm
import torch
import torch.optim as optim
from torch import nn
import time
import copy
import pickle

class AutoML:
    def __init__(self, params: dict):
        self.params = params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_dataset = BatchDataset(params)
        
        splited_dataset = split_dataset(batch_dataset, 42, (80, 20, 0))
        print(f"Train: {splited_dataset['train'].idxs}, \
                Validation: {splited_dataset['valid'].idxs}")

        self.train_loader = DataLoader(splited_dataset["train"], batch_size=params["batch_size"], shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(splited_dataset["valid"], batch_size=params["batch_size"], shuffle=False, num_workers=0)
        self.test_loader = DataLoader(splited_dataset["test"], batch_size=params["batch_size"], shuffle=False, num_workers=0)

        if params["algorithm_name"] == "resnet":
            self.model_ft = models.resnet18(pretrained=True)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, len(params["class_name"]))
        elif params["algorithm_name"] == "alexnet":
            self.model_ft = models.alexnet(pretrained=True)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, len(params["class_name"]))
        elif params["algorithm_name"] == "vgg":
            self.model_ft = models.vgg11_bn(pretrained=True)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, len(params["class_name"]))
        elif params["algorithm_name"] == "densenet":
            self.model_ft = models.densenet121(pretrained=True)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, len(params["class_name"]))

        self.best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        self.best_acc = 0.0
        self.best_loss = 100.0
        self.best_epoch = 0

        self.optimizer = optim.Adam(self.model_ft.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

        if self.device.type == "cuda":
            self.model_ft.cuda()

    def train(self, epoch):
        train_running_loss = 0.0
        train_running_corrects = 0
        self.model_ft.train()
        for batched_data in tqdm.tqdm(self.train_loader, desc="train"):
            inputs = batched_data[0].to(self.device)
            labels = batched_data[1].to(self.device)
            labels = labels.max(dim=1).indices
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model_ft(inputs)
                loss = self.loss_fn(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                self.optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.data)

        train_epoch_loss = train_running_loss / len(self.train_loader.dataset)
        train_epoch_acc = train_running_corrects.double() / len(self.train_loader.dataset)
        print(f'train epoch: {epoch + 1} Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}')

        #valid
        valid_running_loss = 0.0
        valid_running_corrects = 0
        self.model_ft.eval()
        for batched_data in tqdm.tqdm(self.valid_loader, desc="val"):
            inputs = batched_data[0].to(self.device)
            labels = batched_data[1].to(self.device)
            labels = labels.max(dim=1).indices

            with torch.set_grad_enabled(False):
                outputs = self.model_ft(inputs)

                _, preds = torch.max(outputs, 1)

            valid_running_loss += loss.item() * inputs.size(0)
            valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(self.valid_loader.dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(self.valid_loader.dataset)
        print(f'val epoch: {epoch + 1} Loss: {valid_epoch_loss:.4f} Acc: {valid_epoch_acc:.4f}')

        if self.params["hpo"].lower() == "accuracy" and valid_epoch_acc > self.best_acc:
            self.best_acc = valid_epoch_acc
            self.best_loss = valid_epoch_loss
            self.best_epoch = epoch + 1
            self.save_best_model(self.params["model_name"])
        elif self.params["hpo"].lower() == "loss" and valid_epoch_loss < self.best_loss:
            self.best_acc = valid_epoch_acc
            self.best_loss = valid_epoch_loss
            self.best_epoch = epoch + 1
            self.save_best_model(self.params["model_name"])
        
        print(f'HPO Settings: {self.params["hpo"]} Best val Acc: {self.best_acc:4f} Loss: {self.best_loss:4f} Epoch: {self.best_epoch} Model: {self.params["model_name"]}')

    def save_best_model(self, model_name):
        self.best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        model_info = {
            "model": self.best_model_wts,
            "algorithm_name": self.params["algorithm_name"],
            "class_num": len(self.params["class_name"])
        }
        with open(f'models/{model_name}', 'wb') as file:
            pickle.dump(model_info, file)