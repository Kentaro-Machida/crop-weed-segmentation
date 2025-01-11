# crop-weed-segmentation

## 環境構築
以下のコマンドでインストールしてください。環境によっては、入らないかもしれないので、その場合は、一つずつインストールしてください。
```
pip install -r requirements.lock
```
主に使用しているライブラリは以下の通り。
* pytorch
* pytorch lightning
* transformers
* mlflow

## Configファイルによる実験設定
実験の設定は、`config.yml`を編集することで変えることができる。主な設定項目はconfig.ymlにコメントとして残しているが、わかりづらいものだけここで説明。
* modeldataset_type: 使用するモデルのタイプを指定。それぞれの詳細設定は、cnnならcnn_setting、patch2dならpatch2d_settingなどで管理している
    * cnn: CNN系統のモデル
    * transformer: Transformer系統のモデル
    * patch-d: 画像を小さく分割して、セグメンテーションをするタイプのモデル
* task: セグメンテーションにおいて、どのクラスをどのラベル値に割り当てるかを決定するパラメータ
    * plant: [0, 1]の2クラス分類。2, 3, 4などの1以上のラベルはすべて1として扱われる
    * crop: [0, 1]の2クラス分類。1以外のラベルがすべて0として扱われる
    * 3_classes: [0, 1, 2]の3クラス分類。3, 4などの2以上のラベルはすべて2として扱われる
    * 5_classes: [0, 1, 2, 3, 4]の5クラス分類

## モデルの学習
```
python train.py
```
学習を始めると、`mlruns/` というディレクトリが作成される。
ここには、MLFlowによる実験の単位`run`が複数格納される。

## 実験データの見方
MLflowのダッシュボードを使用するには、以下のコマンドを実行`mlruns/`ディレクトリの親ディレクトリ（つまり本リポジトリのルートディレクトリ）で以下のコマンドを実行。
```
mlflow ui --port 5000
```
その後、`http://127.0.0.1:5000`にアクセスすると結果を見ることができる。

## 学習済みモデルによる推論
`train.py`で学習したモデルを読み込んで、テストデータを読み込み、マスクおよび、マスクを入力画像にオーバラップさせた画像を出力する。手順は以下の通り。

1. mlflowを起動し、使用したいモデルの`Experiment ID`と`Run ID`を確認
2. `config.yml`の以下の部分に上記IDをコピペ
```yaml
  one_model_config:
    experiment_id: "865153750648612543"  # ダブルクォーテーションで囲む
    run_id: "bcf9d965735f4ec4962da9590d3ecb87"  # ダブルクォーテーションで囲む
```
3. `python predict.py`を実行
4. mlflowを開くと Artifactsというところに`predictions`というディレクトリができており、そこで入出力画像を確認できる。
