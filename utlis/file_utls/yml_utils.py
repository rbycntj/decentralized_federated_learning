import yaml
import os


def read_yaml(father_label, file_path=os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))) + os.sep + "configuration" + os.sep + "configuration.yml"):
    with open(file_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)[father_label]
