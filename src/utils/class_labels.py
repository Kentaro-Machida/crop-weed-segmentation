def tast_to_label_dict(task: str)->dict:
    """
    このリポジトリ全体で使用されるtaskとlabel_dictの対応を返す関数
    Args:
        task (str): crop | plant | all
    Returns:
        dict: 画像の整数とラベルの対応
    """
    if task == "all":
        return {'background': 0, 'crop': 1, 'weed': 2}
    elif task == "plant":
        return {'background': 0, 'plant': 1}
    elif task == "crop":
        return {'background': 0, 'crop': 1}
    else:
        raise ValueError(f"task: {task} is not supported.")
