import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from inspection.inspector import Inspector


class BatchDataset(Dataset):
    def __init__(self, params: dict):
        self.params = params
        self.image_name_list = os.listdir(params["file_path"])
        self.transform = T.Compose([T.ToTensor(), T.Resize((224,224))])
        self.preprocess = Inspector(params["preprocess"], params["class_name"])
    
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        image_name = self.image_name_list[index].split(".")[0]
        class_name = image_name.split(self.params["delimiter"])[self.params["split_class_index"]]
        target = [0] * len(self.params["class_name"])
        
        for i, item in enumerate(self.params["class_name"]):
            if class_name == item:
                target[i] = 1
                break
        preprocessed = self.preprocess.get_roi_img(os.path.join(self.params["file_path"], self.image_name_list[index]), self.params["method"])
        img = self.transform(preprocessed)
        target = torch.tensor(target, dtype=torch.float32)

        return img, target, self.image_name_list[index]