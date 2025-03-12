import yaml


class TrainingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)
