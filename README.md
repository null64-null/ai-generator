# ニューラルネットワークモデル生成ツール
<br>

## 概要
- ニューラルネットワークモデルを生成するためのプログラムの制作
- 画像判定と生成を行うプログラムの制作
<br><br>

## 開発の目的
- 機械学習をアルゴリズムから理解する
- 最終的にサーバー化し、ドラッグアンドドロップなど感覚的な操作でネットワークを構築して簡単に学習モデルを作れるUIも作成し、Webアプリ化する
<br><br>

## ルール
一から理論を理解したいので、機械学習系ライブラリを使用せずnumpyだけで書く
<br><br>

## 使用技術
* 現在
  * Python
* 今後のWebアプリ化
  * Django, Python（バックエンド）
  * React, ts（フロントエンド）
<br><br>

## 詳細
- 画像判定と生成を、それぞれファイル　image_classification.py、image_generation.pyにおいて層の種類、総数などの各種パラメータを自由に設定して学習を行うことができる
- 題材としてmnist手書き文字を用いて、手書き文字の判定と、数字画像の生成を行っている
- 最終的にサーバー化するので、各種パラメータを設定する箇所は、フロントエンドからの通信を受けて設定する場合に備えた書き方をしている。

### 画像判定を行う場合
- 全結合層、畳み込み層の両方を使用可能
- 各種活性化関数（relu, sigmoid, softmaxなど）を使用可能
- 現状では、80~90%以上の精度で手書き数字の画像の判別に成功

### 画像生成を行う場合
- GAN（Generative Adversarial Networks）によって手書き数字画像を生成
- 現状は上手く画像が生成できていないので、Batch Normarizationなど、出力を収束させる施策を施す予定
<br><br><br>


## 実行例（画像判定）
### パラメータ
|  |  |
| ---- | ---- |
| イテレーション（学習回数） | 1000 回 |
| 損失関数 | cross entropy error |
| トレーニングデータ数 | 60000 |
| テストデータ数 | 10000 |
| ネットワーク | 入力（28×28pix）→ 畳み込み層（フィルタサイズ: 7×7、フィルタ数: 5）→ Relu → 全結合層（入力ノード数: 2420, 出力ノード数: 10）→ Softmax → 出力|

### 結果
- 正答率: 96.44%
- 生成されたネットワークモデルでは、以下のような畳み込みフィルタが生成された
- 手書き文字の各部（直線部分、曲線部分、8のような穴のある形状）などに反応するような画像になっているようである
  
<img width="159" alt="1" src="https://github.com/null64-null/ai-generator/assets/127968084/1fe15062-efb8-4faa-8a64-e44e7f5ae894">　<img width="152" alt="2" src="https://github.com/null64-null/ai-generator/assets/127968084/d53a606f-b689-4f7a-ac56-469776900328">　<img width="155" alt="3" src="https://github.com/null64-null/ai-generator/assets/127968084/6c183301-5cc4-43dd-92f3-fa3cdabfda97">　<img width="151" alt="4" src="https://github.com/null64-null/ai-generator/assets/127968084/22d1c9fa-23da-4b8c-bc10-f800dc5a017e">　<img width="151" alt="5" src="https://github.com/null64-null/ai-generator/assets/127968084/960bdb8f-61eb-4a03-aba5-f8963b360e20">

<br><br>


## 今後やりたいことなど
- 画像生成がまだ上手くできていないので、Batch Normarizationを行う層を組み込むなど、出力を収束させるための工夫を加えたいです。
- 目的にも書いたように、このコードを機械学習生成モデルとしてWebアプリ化したいです。
- ツールで作成したモデルを、自分が作る他のプロダクトにたくさん組み込みたいです。
<br><br>

## ライセンス
本リポジトリのライセンスは、 MIT ライセンスの規約に基づいて付与されています
（LICENCE に記載）
<br><br>
