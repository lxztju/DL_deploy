import yaml


def load_yaml_file(yaml_file):
    with open(yaml_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            raise yaml.YAMLError
    return config