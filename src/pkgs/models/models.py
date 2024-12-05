import torch.nn as nn
from collections import OrderedDict


class PatchFCN(nn.Module):
    """1次元のパッチを入力として受け取り、全結合層を通して1次元のパッチを出力するモデル
    入力パッチのサイズはpatch_heigth * patch_width * 3であることを前提としている

    Args:
        patch_height (int): パッチの高さ
        patch_width (int): パッチの幅
        number_of_layers (int): 全結合層の数
    """
    def __init__(self, patch_heigth:int, patch_width:int, number_of_layers:int):
        assert number_of_layers%2 == 0, "number_of_layers have to be even number"
        self._input_dim = patch_heigth * patch_width * 3
        super().__init__()
        self._output_dim = patch_heigth * patch_width
        self.layers = []
        self._input_dims = [self._input_dim]
        for i in range(1, number_of_layers):
            if i <= number_of_layers/2:
                self._input_dims.append(self._input_dim*(2**i))
            elif i > number_of_layers/2:
                self._input_dims.append(self._output_dim*(2**(i-1)))

        for i in range(number_of_layers):
            if i == number_of_layers-1:
                self.layers.append(
                    (f"layer{i+1}", nn.Linear(self._input_dims[i], self._output_dim))
                    )
            else:
                self.layers.append(
                    (f"layer{i+1}", nn.Linear(self._input_dims[i], self._input_dims[i+1]))
                    )
        
        self.layers.append(
            (f"layer{i+1}", nn.Linear(self._input_dims[-1], self._output_dim))
            )
        self.layers = nn.Sequential(OrderedDict(self.layers))
            

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = PatchFCN(
        patch_heigth=32,
        patch_width=32,
        number_of_layers=6
    )
    print(model)
    print(model.__dict__)
    repr(model)
