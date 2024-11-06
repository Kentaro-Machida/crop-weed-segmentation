from pydantic import BaseModel


class ConfigData(BaseModel):
    train_parameter: dict
    data_root: str
    task: str
    plant_model: dict
    crop_model: dict
    all_model: dict


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


if __name__ == '__main__':
    all_model_data = AllModelData(modeltype="segformer", backborn="mobilenet_v2")
    print(all_model_data)
