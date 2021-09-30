from train import *
import argparse, os
import consts, utils

def main():
    model_info = utils.read_json_file(args.model_info)
    preprocess_params = utils.read_json_file(args.params)
    dataset = os.listdir(args.data_dir)

    train_params = {
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "class_name": model_info["class"]["sound"],
        "file_path": args.data_dir,
        "hpo": args.hpo,
        "split_class_index": consts.SPLIT_CLASS_INDEX,
        "delimiter": consts.DELIMITER,
        "algorithm_name": args.algorithm,
        "model_name": args.model_name,
        "preprocess": preprocess_params,
        "method": args.method
    }

    trainer = AutoML(train_params)
    for epoch in range(args.epoch):
        try:
            trainer.train(epoch)
        except Exception as e:
            print(f"train: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract mel spectrogram from wave, train them, test the trained model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help="dataset for test")
    parser.add_argument('--params', type=str, default=consts.PARAMETER_PATH,
                        help="parameter path for preprocessing")
    parser.add_argument('--method', type=str, default="extract_mel_features",
                        help="select any preprocess method in parameters.json file")
    parser.add_argument('--model_info', type=str, default=consts.MODEL_INFO_PATH,
                        help="information of model including class types")
    parser.add_argument('--algorithm', type=str, default="resnet",
                        help="choice: resnet, alexnet, vgg, densenet")
    parser.add_argument('--epoch', type=int, default=10,
                        help="number of total epochs to run")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--hpo', type=str, default="accuracy",
                        help="choice: accuracy, loss (what metric will be prioritized when saving a model)")
    parser.add_argument('--model_name', type=str, default="resnet_model",
                        help="model name to save")

    args = parser.parse_args()
    main()