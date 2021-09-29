import json, csv
from inspection.preprocess_fns import Preprocess
from inspection.inference import Inference
from sklearn import metrics

class Inspector():
    def __init__(self, params: dict, classes: list):
        super().__init__()
        self.inference = Inference()
        self.preprocess = Preprocess()
        self.initialized_model = ""
        self.params = params
        self.classes = classes

    def ai_initialize(self, model_name):
        if self.initialized_model != model_name:
            self.inference.initialize(model_name)    
            self.initialized_model = model_name
            return True
        return False

    def get_roi_img(self, input, method):
        if method == "extract_mel_features":
            gray_img = self.preprocess.extract_mel_features(input, self.params["extract_mel_features"])
            return self.preprocess.gray_to_rgb(gray_img)
        elif method == "extract_signal_features":
            gray_img = self.preprocess.extract_signal_features(input, self.params["extract_signal_features"])
            return self.preprocess.gray_to_rgb(gray_img)

    def evaluate(self, input, method):
        roi_img = self.get_roi_img(input, method)
        class_num = self.inference.evaluate(roi_img)
        class_name = self.__convert_num_to_name(class_num)
        return class_name

    def evaluate_batch(self, imgs):
        class_num_list = self.inference.evaluate_batch(imgs)
        class_name_list = []
        for class_num in class_num_list:
            class_name_list.append(self.__convert_num_to_name(class_num))
        return class_name_list

    def save_json_output(self, result_ls:list):
        json_output = {}
        for class_name in self.classes:
            json_output[class_name] = []

        for value in result_ls:
            image_name = value[0]
            class_name = value[1]
            json_output[class_name].append(image_name)

        with open("./result.json", 'w') as f:
            json.dump(json_output, f, indent=4)

    def save_output_csv(self, result_ls:list, metric_ls:list):
        with open('./result.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for item in metric_ls:
                writer.writerow(item)
            for item in result_ls:
                writer.writerow(item)

    def get_metrics(self, result_ls:list):
        y_true = []
        y_pred = []
        for res in result_ls:
            label = res[1]
            pred = res[2]
            y_true.append(self.__convert_name_to_num(label))
            y_pred.append(self.__convert_name_to_num(pred))

        merics_result = {
            "accuracy": {},
            "f1_score": {},
            "precision": {},
            "recall": {},
        }
        for met in merics_result:
            merics_result[met]["average"] = 0

        if not y_true or not y_pred:
            return

        # metric by class
        precision_ls = metrics.precision_score(y_true, y_pred, average=None, zero_division=1)
        recall_ls = metrics.recall_score(y_true, y_pred, average=None, zero_division=1)

        class_index = 0
        for i, class_name in enumerate(self.classes):
            if not i in y_pred:
                continue
            class_name = self.__convert_num_to_name(i)
            merics_result["precision"][class_name] = round(precision_ls[class_index], 2)
            merics_result["recall"][class_name] = round(recall_ls[class_index], 2)
            class_index += 1

        # Average
        merics_result["accuracy"]["average"] = round(metrics.accuracy_score(y_true, y_pred), 2)
        merics_result["f1_score"]["average"] = round(metrics.f1_score(y_true, y_pred, average="macro", zero_division=1), 2)
        merics_result["precision"]["average"] = round(sum(merics_result["precision"].values()) / len(self.classes), 2)
        merics_result["recall"]["average"] = round(sum(merics_result["recall"].values()) / len(self.classes), 2)
        
        return merics_result

    def get_ratio(self, num, total_num):
        return round(num / total_num if total_num > 0 else 0.0, 3) * 100

    def __convert_num_to_name(self, class_num):
        for i, item in enumerate(self.classes):
            if class_num == i:
                return item

    def __convert_name_to_num(self, class_name):
        for i, item in enumerate(self.classes):
            if class_name == item:
                return i