# Image Compression Project

面向图像压缩应用的“传统编码器统一接口 + 自适应策略选择”基线项目。当前版本只包含 JPEG、JPEG 2000、JPEG XL、AVIF、BPG 等传统编码器封装，以及基于图像特征和指标约束的策略选择；不包含 CompressAI、PyTorch、神经网络压缩或模型训练。

## 支持的编码器

| Codec | 实现方式 | 备注 |
|---|---|---|
| JPEG | Pillow | 默认 RGB、4:4:4、`optimize=True` |
| JPEG 2000 | OpenJPEG `opj_compress` / `opj_decompress` | 自动检测工具是否存在 |
| JPEG XL | libjxl `cjxl` / `djxl` | 使用 `distance` 参数 |
| AVIF | libavif `avifenc` / `avifdec` | 默认尝试 `--yuv 444`，失败后回退默认设置 |
| BPG | `bpgenc` / `bpgdec` | 可选；没有工具时自动跳过 |

## 安装

Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ubuntu 外部工具示例：

```bash
sudo apt-get update
sudo apt-get install libjxl-tools libavif-bin openjpeg-tools
```

BPG 通常不在默认源中，属于可选编码器。程序会通过 `shutil.which()` 检测 `cjxl`、`djxl`、`avifenc`、`avifdec`、`opj_compress`、`opj_decompress`、`bpgenc`、`bpgdec`，不可用时记录 warning 并跳过该编码器。

## 单图压缩

```bash
python compress.py \
  --input data/kodak/kodim01.png \
  --output-dir outputs/kodim01 \
  --mode ratio_first \
  --target-ratio 16 \
  --min-psnr 35 \
  --codecs jxl,avif,jpeg2000 \
  --search exhaustive
```

输出结构示例：

```text
outputs/kodim01/
  candidates/
    jxl_distance_1p5_effort_7/
      compressed.jxl
      recon.png
      result.json
  best/
    compressed.jxl
    recon.png
    best_result.json
  all_results.csv
  features.json
```

## Kodak 批量测试

```bash
python eval.py \
  --input-dir data/kodak \
  --output-dir reports/kodak_eval \
  --mode ratio_first \
  --target-ratio 16 \
  --min-psnr 35 \
  --codecs jpeg,jxl,avif,jpeg2000 \
  --recursive false
```

## 遥感图像批量测试

```bash
python eval.py \
  --input-dir data/remote \
  --output-dir reports/remote_eval \
  --mode remote_sensing \
  --target-ratio 16 \
  --min-psnr 35 \
  --codecs jxl,avif,jpeg2000 \
  --recursive true
```

批量评估会生成：

```text
per_image_results.csv
all_candidate_results.csv
summary.json
summary.md
rd_scatter.png
psnr_hist.png
cr_hist.png
```

## 指标定义

- `CR`：压缩倍数，`原始理论大小 / 压缩码流大小`。
- `原始理论大小`：对 8-bit RGB 图像使用 `H * W * 3` bytes，不使用 PNG/JPEG/TIFF 文件在磁盘上的大小。
- `bpp`：`压缩码流大小 * 8 / (H * W)`。
- `PSNR`：RGB 三通道整体 MSE，`MAX_I = 255`；完全一致时返回 `inf`。
- `SSIM`：`skimage.metrics.structural_similarity`，`channel_axis=-1`，`data_range=255`。
- `MS-SSIM`：当前作为可选指标保留接口，默认返回 `None`，不会导致评估失败。
- `Edge-PSNR`：用 Canny 边缘 mask 只在边缘区域计算 PSNR；没有边缘时回退普通 PSNR。

## 自适应策略

- `ratio_first`：先要求 `CR >= target_ratio`，在满足压缩率的候选中选择 PSNR 最高者。若无候选达到目标压缩率，选择最接近目标压缩率且 PSNR 更高的结果，并记录 warning。
- `quality_first`：先要求 `PSNR >= min_psnr`，在满足质量的候选中选择 CR 最高者。若无候选达到最低 PSNR，选择 PSNR 最高的结果，并记录 warning。
- `remote_sensing`：先要求 `CR >= target_ratio`，再按 PSNR、SSIM、Edge-PSNR、CR 的加权分数选择。边缘密度较高时提高 Edge-PSNR 权重，降低 CR 权重。

`passed` 表示最终结果同时满足验收目标：`CR >= target_ratio` 且 `PSNR >= min_psnr`。

## 图像读取约定

支持 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff`、`.bmp`。所有输入读取后统一转为 RGB `uint8`：

- 灰度图会扩展为 RGB。
- RGBA 会丢弃 alpha 并转为 RGB。
- Kodak 图像不 resize、不裁剪，保持原图尺寸。

## 参数方向说明

- JPEG / AVIF 的 `quality` 越大质量越高、文件越大。
- JPEG XL 的 `distance` 越小质量越高、文件越大。
- JPEG 2000 的 `rate` 表示目标压缩倍数。
- BPG 的 `quality`/`q` 越小质量越高、文件越大，这一点和 JPEG 相反。

## 测试

```bash
source .venv/bin/activate
pytest tests -q
```

## 注意事项

项目目录按需求包含 `codecs/`，它与 Python 标准库 `codecs` 同名。运行入口和测试会先启用本地子模块路径，以便 `codecs/base.py` 等文件可正常导入，同时保留标准库 `codecs` 的原有功能。
