# crop-weed-segmentation

## モデルの学習
```
python train.py
```
学習を始めると、`mlruns/` というディレクトリが作成される。
ここには、MLFlowによる実験の単位`run`が複数格納される。

## 実験データの見方
MLflowのダッシュボードを使用するには、以下のコマンドを実行
```
cd /mlruns
mlflow ui --port 5000
```
その後、`http://127.0.0.1:5000`にアクセスすると結果を見ることができる。