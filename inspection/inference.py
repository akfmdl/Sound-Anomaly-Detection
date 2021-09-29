from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image  # pytorch grad cam
                                                            # https://webcache.googleusercontent.com/search?q=cache:fwsU-G-9arwJ:https://github.com/jacobgil/pytorch-grad-cam+&cd=1&hl=en&ct=clnk&gl=kr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import pickle

class Inference():
    def initialize(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(model_path, 'rb') as file:
            model_info = pickle.load(file)

        if model_info["algorithm_name"] == "resnet":
            self.model_ft = models.resnet18(pretrained=True)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, model_info["class_num"])
        elif model_info["algorithm_name"] == "alexnet":
            self.model_ft = models.alexnet(pretrained=True)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, model_info["class_num"])
        elif model_info["algorithm_name"] == "vgg":
            self.model_ft = models.vgg11_bn(pretrained=True)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, model_info["class_num"])
        elif model_info["algorithm_name"] == "densenet":
            self.model_ft = models.densenet121(pretrained=True)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, model_info["class_num"])


        self.model_ft.load_state_dict(model_info["model"])
        if self.device.type == "cuda":
            self.model_ft = self.model_ft.cuda() 
        self.model_ft.eval()
        self.preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            #Normalize(mean=mean, std=std)
        ])

        # TODO: 첫 이미지만 속도가 느려서 초기화시 한번 호출하도록 조치(Pytorch 1.9버전에서 문제발생). 추후 version up 되면 수정
        np_array = np.zeros(shape=(1, 1, 3), dtype="uint8")
        input_tensor = self.preprocessing(np_array).unsqueeze(0)
        output = self.model_ft(input_tensor.cuda()) if self.device.type == "cuda" else self.model_ft(input_tensor)
        print("------------------------------------- AI initialized successfully -------------------------------------")

    def evaluate(self, img):        
        input_tensor = self.preprocessing(img).unsqueeze(0)
        output = self.model_ft(input_tensor.cuda()) if self.device.type == "cuda" else self.model_ft(input_tensor)
        return int(output.max(dim=1).indices)

    def evaluate_batch(self, imgs):
        imgs = imgs.to(self.device)
        output = self.model_ft(imgs.cuda()) if self.device.type == "cuda" else self.model_ft(imgs)
        return output.max(dim=1).indices

    def get_heatmap_data(self, img):
        input_tensor = self.preprocessing(img).unsqueeze(0)

        output = self.model_ft(input_tensor.cuda()) if self.device.type == "cuda" else self.model_ft(input_tensor)
        output = int(output.max(dim=1).indices)

        target_layer = self.model_ft.layer4[-1]
        use_cuda = True if self.device.type == "cuda" else False
        cam = GradCAM(model=self.model_ft, target_layer=target_layer, use_cuda=use_cuda)
        target_category = None
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        result_img = show_cam_on_image(img, grayscale_cam)

        return result_img, output