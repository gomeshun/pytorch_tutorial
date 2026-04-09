# PyTorch Tutorial Collection

`uv` で管理する、PyTorch によるチュートリアル集です。`tutorial_mnist.ipynb` では `torchvision.datasets.MNIST` を使ったシンプルな畳み込みニューラルネットワークの学習を扱い、あわせて線形回帰と Mixture Density Network のノートブックも含めています。

現在の推奨構成は Python 3.14 + `torch==2.10.0` + `torchvision==0.25.0` です。Linux ではこの組み合わせで CUDA 12.8 系が解決されるため、CUDA 13 系を避けられます。

## Requirements

- `uv`
- Python 3.14

## Setup

```bash
uv sync
```

依存関係の同期後、`.venv` が作成されます。ノートブックではこの `.venv` を Python カーネルとして選択してください。MNIST データセットは初回実行時に `data/` 以下へ自動ダウンロードされます。

## Notebooks

- `tutorial_mnist.ipynb`: MNIST 分類のエンドツーエンド例
- `tutorial_linear_regression.ipynb`: 基本的な回帰モデルの導入
- `tutorial_mdn.ipynb`: Mixture Density Network による条件付き分布の学習

各ノートブックは self-contained なので、カーネルに `.venv` の Python 3.14 を選び、上から順にセルを実行すれば学習ループやモデル定義を追えます。

## Run

```bash
uv run mnist-train --epochs 3 --batch-size 128
```

主なオプション:

- `--learning-rate`: 学習率
- `--data-dir`: データ保存先
- `--num-workers`: DataLoader の worker 数
- `--seed`: 乱数シード

## Project Layout

```text
.
|-- pyproject.toml
|-- README.md
|-- tutorial_linear_regression.ipynb
|-- tutorial_mdn.ipynb
|-- tutorial_mnist.ipynb
|-- .vscode/tasks.json
`-- src/mnist_tutorial/
    |-- data.py
    |-- model.py
    `-- train.py
```

## What The Script Does

1. MNIST の訓練・テストデータをダウンロードして正規化
2. 小さな CNN を定義
3. Cross Entropy Loss と Adam オプティマイザで学習
4. 各 epoch の損失と精度を表示