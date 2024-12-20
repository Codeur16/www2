import json
from pathlib import Path
import pandas as pd


def convert2submit(test_file: Path, prediction_file: Path, save_path: Path):
    pred_label_list = []

    for line in open(prediction_file, "r"):
        prediction_data = json.loads(line)

        pred_label = prediction_data["predict"]
        pred_label_list.append(pred_label)

    test_data = json.load(open(test_file, "r"))
    save_data = []
    for i, example in enumerate(test_data):
        example["predict"] = pred_label_list[i]
        save_data.append(example)

    df = pd.DataFrame(save_data)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")


if __name__ == "__main__":
    test_file = "data/demo_test.json"
    prediction_file = "data/demo_pred.jsonl"
    save_path = "submit.csv"
    convert2submit(test_file, prediction_file, save_path)

# end main
