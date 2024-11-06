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


if __name__ == '__main__':
    all_model_data = AllModelData(modeltype="segformer", backborn="mobilenet_v2")
    print(all_model_data)
