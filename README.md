# CoDiRec

**共起バイアスと方向バイアスの明示的注入による逐次推薦モデル**
*(Explicit injection of co-occurrence and directional priors for sequential recommendation)*

逐次推薦において、(1) **共起バイアス**（対称：どのアイテムが一緒に出るか）と (2) **方向バイアス**
（反対称：A の後に B が来やすい precedence）は、データに強く存在し予測的でありながら、標準的な
モデルには暗黙的・不完全にしか取り込まれない。本リポジトリは、この2つの関係構造を **学習データから
構築した事前情報として明示的に注入**する逐次推薦モデル **CoDiRec** を提供する。

## モデル構成 (`model/codirec.py`)

バックボーン = FFT 周波数フィルタ → 双方向 Mamba ∥ 自己注意 をゲート融合 → GLU。
そこに2つの事前情報を **対称性に応じて役割整合的に**注入する：

- **共起 C（対称, PPMI/L2）→ 自己注意への加法ペアワイズバイアス**
  （注意は順序不問・ペアワイズ＝対称関係の自然な置き場）
  `scores = QKᵀ/√d + mask + α_co · C[ids, ids]`
- **方向 D（反対称, `(f_ij−f_ji)/(f_ij+f_ji)`）→ Mamba 前向きスキャンの入力変調**
  （因果・方向＝非対称な遷移の置き場）
  `x_fwd = x · (1 + dir_gate · D[item_{i-1}, item_i])`

`α_co`, `dir_gate`, 融合ゲートは **0 で初期化** → 初期状態は素のバックボーンと等価で、学習で事前情報の
使い方を獲得する。損失は全語彙クロスエントロピー。C, D は **学習区間 `s[:-2]` のみ**から構築（リーク安全）。

### （オプション）系列拡張 `--seq_augment`
入力系列に共起＋方向の関連アイテムを挿入する拡張 (`model/cooc_dir.build_augment_associates`)。
本研究の実験では精度向上に寄与しなかったため既定オフ（再現用に同梱）。

## ファイル構成
```
main.py / trainers.py / utils.py / dataset.py / metrics.py   # 学習・評価パイプライン
model/
  codirec.py   # 提案モデル CoDiRec（本体）
  cooc_dir.py           # 共起 C・方向 D 行列／系列拡張の関連アイテム構築
  echomamba4rec.py      # バックボーン部品（FilterLayer, GLU 等）
  _abstract_model.py / _modules.py
```
※ ベースラインは含まない。データは同梱せず **BSARec のデータセットを参照**する。

## 必要環境
- Python 3.10, PyTorch (CUDA), `mamba-ssm`（CUDAカーネル必須・CPU不可）, scipy, numpy, tqdm
- **データ**：[BSARec](https://github.com/yehjin-shin/BSARec) のデータセットを参照。本リポジトリを
  BSARec と同じ親ディレクトリに置く（`../BSARec/src/data/` が既定）か、`--data_dir` で指定する。

## 実行例（各データセットの best HP）
```bash
# ML-1M
python main.py --model_type CoDiRec --train_name codir_ml1m --data_name ML-1M \
  --lr 0.001 --d_state 16 --num_hidden_layers 2 --hidden_dropout_prob 0.2 --attention_probs_dropout_prob 0.2 \
  --d_conv 4 --expand 2 --hidden_size 64 --num_attention_heads 1 \
  --batch_size 256 --epochs 200 --patience 10 --seed 42 \
  --codir_norm l2 --codir_dir_norm ratio_l2 --codir_cap 50 --codir_window 5

# Beauty : --lr 0.0005 --d_state 64 --num_hidden_layers 1 --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.5
# LastFM : --lr 0.001  --d_state 16 --num_hidden_layers 1 --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.5
```
評価のみ：`--do_eval --load_model <train_name>`。ログ/チェックポイントは `output/` に出力。

## 謝辞・ライセンス
本実装は **BSARec** (Shin et al., AAAI 2024, https://github.com/yehjin-shin/BSARec) の
フレームワークを基盤として拡張したもの。データセットおよびパイプラインの一部は BSARec に由来する。
利用の際は BSARec のライセンス・規約に従うこと。
