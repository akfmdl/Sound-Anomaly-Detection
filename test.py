from inspection import *
import argparse, os
import consts, utils

def main():
    model_info = utils.read_json_file(args.model_info)
    params = utils.read_json_file(args.params)
    dataset = os.listdir(args.data_dir)

    inspector = Inspector(params, model_info["class"]["sound"])
    inspector.ai_initialize(args.model_path)

    # preprocess + AI inspection
    result_ls = []
    for data in dataset:
        try:
            class_name = inspector.evaluate(os.path.join(args.data_dir, data), args.method)
            split_name = data.split(consts.DELIMITER)
            label = split_name[1] if len(split_name) > 2 else ""
            result_ls.append([data, label, class_name])
        except Exception as e:
            print(f"test: {str(e)}")

    metric_dict = inspector.get_metrics(result_ls)
    metric_ls = utils.dict_to_list(metric_dict)
    inspector.save_output_csv(result_ls, metric_ls)

if __name__ == '__main__':
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
    parser.add_argument('--model_path', type=str, default=consts.MODEL_PATH+"/default",
                        help="model path for using in this test")

    args = parser.parse_args()
    main()