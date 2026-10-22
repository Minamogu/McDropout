# **第4回 実験計画**
2026年10月15日

## 1. 実験概要
健常者および肺炎患者の胸部X線画像データセットを用い、Dropoutを含むCNNモデルを構築する。\
MC Dropoutを用いて予測の不確実性を定量化し、誤分類検出能力及び予測確率の校正性能を評価することで、不確実性推定手法の有効性を検証する。
本実験ではチャネル数やdropout率といったモデルの構造を変更し、各評価指標に生じる変化を比較することでこれらの設定が不確実性推定性能に与える影響を検証する。

## 2. データセット
- 使用データ: PneumoniaMNIST (肺炎)
- データの分割: train:4700, validation:500, test:600


## 3. モデル仕様
### 3.1 モデルアーキテクチャ

以下のCNNアーキテクチャはPyTorchで実装する。

```
【Input】
- Input image size: 128 × 128 × 1
- Grayscale image
- Binary classification

【Block 1】
- Conv2d(in_channels=1, out_channels=C1, kernel_size=3, stride=1, padding=1)
- BatchNorm2d(C1)
- ReLU
- MaxPool2d(kernel_size=2, stride=2)
- Output feature map: 64 × 64 × C1
- Dropout(p)

【Block 2】
- Conv2d(in_channels=C1, out_channels=C2, kernel_size=3, stride=1, padding=1)
- BatchNorm2d(C2)
- ReLU
- MaxPool2d(kernel_size=2, stride=2)
- Output feature map: 32 × 32 × C2
- Dropout(p)

【Block 3】
- Conv2d(in_channels=C2, out_channels=C3, kernel_size=3, stride=1, padding=1)
- BatchNorm2d(C3)
- ReLU
- MaxPool2d(kernel_size=2, stride=2)
- Output feature map: 16 × 16 × C3
- Dropout(p)

【Fully Connected Head】
- Flatten
- Input dimension: 16 × 16 × C3
- Linear(16 × 16 × C3, N1)
- ReLU
- Dropout(p)
- Linear(N1, 1)
- Output: one logit
```

#### 【Training】
- 損失関数にはBCEWithLogitsLossを使用する。
- モデル内部ではSigmoid関数を適用しない。
- 最終層の出力は1つのlogitとする。

#### 【Prediction】
- 推論時には、最終層から得られたlogitにSigmoid関数を適用し、0〜1の予測確率に変換する。
- 予測確率は以下のように計算する。
 probability = sigmoid(logit)

#### 【MC Dropout】
- このモデルは、Monte Carlo  Dropout（MC Dropout）による不確実性推定に使用する。
- 通常の学習時はDropoutを有効にする。
- 通常の評価時はDropoutを無効にする。
- MC Dropoutによる推論時には、Dropoutを有効にしたまま推論する。
- MC Dropout推論時には、Batch Normalizationは評価モード（eval()）を維持する。
- 同じ入力画像に対して、Dropoutの確率的な挙動を利用して30回の確率的forward passを行う。
- 各forward passから予測確率を取得し、30個の予測確率を得る。
- 得られた30個の予測結果を用いて、不確実性指標を計算できるようにする。
- MC Dropoutの推論回数はすべての実験条件でT=30に固定する。

#### 【実装上の条件】
- CNNモデルはPyTorchの nn.Module を継承して実装する。
C1, C2, C3, N1、およびDropout率 p はコンストラクタから変更できるようにする。
- ベースラインの設定は C1=32, C2=64, C3=128, p=0.3, N1=128 とする。
- MC Dropout推論時にDropoutのみを有効化し、Batch Normalizationは評価モードに維持できるように実装する。

### 2.2 ハイパーパラメータ
- batch size: 64
- epochs: 100
- lr: 0.0001
- optimizer: Adam
- early_stopping:
  monitor: validation_loss
  mode: min
  patience: 4
  min_delta: 0.001
  restore_best_weights: true
- weight_decay: 1e-4

## 4. 実験方法
本実験では上記のCNNモデルをベースラインとし、チャネル数及びDropout率をそれぞれ変更した比較実験を行う。
各モデルに対してMC Dropoutの推論をT=30で行い、それぞれの評価指標を算出する。

### 4.1 実験1 チャネル数による比較
CNNのチャネル数の違いが分類性能および不確実性推定性能に与える影響を検証する。
Dropout率をp=0.3に固定し、３つの畳み込みブロックのチャネル数を以下の3条件に変更する。
また、それぞれの条件において3種類の各シード値ごとに評価指標を出し、3シード間の平均値および標準偏差を算出する。

| 条件 | C1 | C2 | C3 |
| ------ | ---- | ---- | ---- |
| A | 16 | 32 | 64 |
| B(ベースライン) | 32 | 64 | 128 |
| C | 64 | 128 | 256 |


- 全結合層のユニット数: N1=128
- 乱数シード: 5, 38, 42
- 学習回数: 3条件 × 3シード = 9回

### 4.2 実験2
Dropout率の違いが分類性能および不確実性推定性能に与える影響を検証する。
チャネル数をベースライン（C1=32, C2=64, C3=128）に固定し、モデル内のすべてのDropout層に対して以下の3条件を設定する。
また、それぞれの条件において3種類の各シード値ごとに評価指標を出し、3シード間の平均値および標準偏差を算出する。

| 条件 | Dropout率 |
| ---- | ---- |
| A | 0.1 |
| B(ベースライン) | 0.3 |
| C | 0.5 |

- 全結合層のユニット数: N1=128
- 乱数シード: 5, 38, 42
- 学習回数: 3条件 × 3シード = 9回

## 5. 評価方法
### 評価指標
- accuracy/loss
- predictive entropy: T回の予測確率の平均値を最終的な予測確率とし、それ対して二値エントロピーを計算する。
  $$
  \bar{p} = \frac{1}{T}\sum_{t=1}^{T}p_t
  $$
  $$
  H(\bar{p}) = -\bar{p}\log(\bar{p}) - (1-\bar{p})\log(1-\bar{p}))
  $$
- Mutual Information (MI): 平均予測確率のPredictive Entropyから、T回の各予測確率に対するエントロピーの平均値を引いて算出する。
  $$
  H(p_t) = -p_t\log(p_t) - (1-p_t)\log(1-p_t)
  $$
  $$
  MI = H(\bar{p}) - \frac{1}{T}\sum_{t=1}^{T}H(p_t)
  $$
- ECE: 10個の等幅便を使用して計算する。各サンプルの予測クラスは平均予測確率が0.5以上の場合を陽性とする。予測のクラスに対するconfidenceと実際のaccuracyを各ビンで計算し、その差の絶対値をサンプル数で重み付けして合計する。
- Error Detection auroc: 正分類を0、誤分類を1とし、Predictive EntropyおよびMutual Information（MI）をそれぞれ不確実性スコアとしてError Detection AUROCを算出する。

### グラフ・画像
- 学習曲線（loss/accuracyの推移）
- error detection auroc: 正解・不正解をラベルとして不確実性スコアで誤分類を検出するAUROC
- 正分類・誤分類の不確実性ヒストグラム

## 6. 実行環境
- GPU: NVIDIA RTX 3060
- Visual Studio Code
- OS:
- Python=
- フレームワーク: Pytorch

