from pydantic import BaseModel


class TrainParameterData(BaseModel):
    max_epocks: int
    device: str
    batch_size: int


class PlantModelData(BaseModel):
    patch_size: int
    modeltype: str


class CropModelData(BaseModel):
    modeltype: str
    backborn: str


class AllModelData(BaseModel):
    modeltype: str
    backborn: str


class ConfigData(BaseModel):
    train_parameter: TrainParameterData
    data_root: str
    task: str
    resized_image_height: int
    resized_image_width: int
    plant_model: PlantModelData
    crop_model: CropModelData
    all_model: AllModelData

    @classmethod
    def load_data(cls, data: dict):
        train_parameter = TrainParameterData(**data["train_parameter"])
        plant_model = PlantModelData(**data["plant_model"])
        crop_model = CropModelData(**data["crop_model"])
        all_model = AllModelData(**data["all_model"])
        return cls(
            train_parameter=train_parameter,
            data_root=data["data_root"],
            task=data["task"],
            resized_image_height=data["resized_image_height"],
            resized_image_width=data["resized_image_width"],
            plant_model=plant_model,
            crop_model=crop_model,
            all_model=all_model
        )


if __name__ == '__main__':
    all_model_data = AllModelData(modeltype="segformer", backborn="mobilenet_v2")
    print(all_model_data)
    config = ConfigData.load_data({
        "train_parameter": {
            "max_epocks": 10,
            "device": "gpu",
            "batch_size": 8
        },
        "data_root": "path/to/data",
        "task": "all",
        "resized_image_height": 512,
        "resized_image_width": 512,
        "plant_model": {
            "patch_size": 32,
            "modeltype": "cnn"
        },
        "crop_model": {
            "modeltype": "segformer",
            "backborn": "mobilenet_v2"
        },
        "all_model": {
            "modeltype": "segformer",
            "backborn": "mobilenet_v2"
        }
    })
    print(config)
