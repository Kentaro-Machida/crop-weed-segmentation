from dataclasses import dataclass

@dataclass
class config_data_class:
    train_parameter: dict
    data_root: str
    task: str
    plant_model: dict
    crop_model: dict
    all_model: dict


@dataclass
class train_parameter_data_class:
    max_epocks: int
    device: str
    batch_size: int


@dataclass
class plant_model_data_class:
    patch_size: int
    model_type: str


@dataclass
class crop_model_data_class:
    model_type: str
    backborn: str


@dataclass
class all_model_data_class:
    model_type: str
    backborn: str

