# 高效率的 BERT 古文断句标点

本作业包括中文预训练 BERT 的全参数微调，以及超过 512 token 的文本如何通过重叠窗口和 Batch 高效解码。训练和定量测试采用老师提供的 1998 年上半年人民日报，符合题目允许使用现代文训练和测试的说明。古文示例只展示推理流程，不能替代古文测试集的准确性评估。


## 已完成的实验

本次在单张 NVIDIA A100 SXM4 80GB 上完成中文 BERT 全参数微调，训练 3 轮、2,256 次实际优化器更新，无溢出跳过。按验证集选择第 3 轮模型。结果均来自实际执行，详细证据见 `runs/main/`。

| 测试指标 | 逐窗基线 | 固定窗口 Batch 32 |
| --- | ---: | ---: |
| 标点 micro F1（排除 NONE） | 78.87% | 78.91% |
| 标点 macro F1 | 64.02% | 63.91% |
| 句界 F1 | 84.73% | 84.84% |

完整测试集含 3,207 篇文章、1,858,379 个正文字符。速度实验使用其中最长的 40 篇文章、166,601 个字符，预热后重复三次取中位数。

| 方法 | 中位耗时 秒 | 字符每秒 | 相对逐窗加速 |
| --- | ---: | ---: | ---: |
| 逐窗 B1 | 4.725 | 35,258 | 1.00× |
| 固定 B1 | 4.391 | 37,944 | 1.08× |
| 固定 B4 | 1.310 | 127,210 | 3.61× |
| 固定 B8 | 0.755 | 220,557 | 6.26× |
| 固定 B16 | 0.616 | 270,370 | 7.67× |
| 固定 B32 | 0.567 | 293,808 | 8.33× |
| 固定 B64 | 0.541 | 308,120 | 8.74× |

现代文训练测试的精度不能等同于古文精度。`examples/` 的古文样例展示实际推理输出，不作为古文准确率测量。

## 目录

```text
punct/data.py        语料清洗、按文章划分、字符后标点标签
punct/train.py       全参数微调、验证集选择模型
punct/engine.py      字符对齐、固定窗口 Batch 解码、逐窗基线
punct/metrics.py     非 NONE 标点 P/R/F1、分类别与断句指标
punct/evaluate.py    整篇文档测试、速度与显存比较
punct/predict.py     任意长度文本推理入口
tests/              真实小型 BERT 前向及边界测试
data/stats.json     语料统计、编码和原始文件 SHA-256
runs/main/          实际运行配置、训练日志、模型哈希与测试结果
```

## 安装和复现

建议使用 Python 3.12。先安装适合本机 CUDA 的 PyTorch，再安装其余依赖。CPU 可运行推理和小规模检查，完整训练建议使用 NVIDIA GPU。

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

本仓库包含代码和已完成实验的结果记录，原始语料与模型权重不纳入 Git。使用老师提供的 `1998-上半年-带音(1).rar`，先解压，再生成训练、验证与测试数据：

```bash
python -m punct.data --source "/path/to/extracted/corpus" --output data
```

使用固定版本的官方预训练模型训练；正式训练会执行三轮，每轮在完整验证集评估，按非 NONE 标点 micro F1 选择最佳模型。输出目录已有模型时拒绝覆盖。

```bash
python -m punct.train --data data --model google-bert/bert-base-chinese --revision 8f23c25b06e129b6c986331a13d8d025a92cf0ea --output runs/reproduce --epochs 3 --batch-size 32 --max-chars 400
python -m punct.evaluate --model runs/reproduce/best --data data/test.jsonl --output runs/reproduce/evaluation --batch-size 32 --batches 1,4,8,16,32,64 --repeats 3
```

完成上面的训练后，使用生成的最佳模型推理：

```bash
python -m punct.predict --model runs/reproduce/best --text "学而时习之不亦说乎有朋自远方来不亦乐乎"
python -m punct.predict --model runs/reproduce/best --input your_long_text.txt --output prediction.txt --batch-size 16
```

`--device cpu` 可在没有 NVIDIA GPU 的电脑运行。输入与输出保持原来的正文字符；分词空格、注音和既有标点会先移除，再预测 7 种支持的标点。换行按段落保留。

## 数据和指标口径

1—4 月用于训练、5 月验证、6 月测试。先按 `日期-版面-文章` 聚合，段落末尾编号不进入模型。全语料按去标点文章正文做精确去重，避免相同正文跨集合；这不保证语义近似报道完全不重叠。含损坏编码的文章整篇剔除。具体统计见 `data/stats.json`。

每个正文字符对应一个模型输入 token，后面对应 NONE、逗号、句号、问号、叹号、分号、冒号或顿号。数字和拉丁字母逐字符编码，模型输入中的全角兼容字符做单字符规范化，输出保留原字形。生僻字使用 UNK；引号、书名号、括号、破折号和省略号不在标签集合中，会被删除。这是本作业明确的任务边界。

主要指标为排除 NONE 的标点 micro F1，标点位置和类别同时正确才计 TP。另报 7 种标点的 macro F1、逐类指标及把句号/问号/叹号合并后的句界 F1。字符准确率容易被大量 NONE 抬高，不作为主要结论。

## Batch 长文本算法

固定窗口的核心区长度为 300 字符，左右各加最多 50 字符上下文，总长度至多 400；加 CLS/SEP 后至多 402 token，小于 512。窗口位置只由原文长度决定，因此可提前生成并跨文章组成 Batch。

每个字符只由它所属的一个核心区输出标签。相邻窗口的上下文不重复写回，因此不存在拼接时重复字、漏字或重复插入标点。最后不足 300 字符的尾部也会输出。中间窗口同时获得左、右上下文，窗口策略的效果由独立测试结果比较。

逐窗基线每次读 400 字符，在前 300 字符最后一个预测句界结束；若句界过早或没有句界则前进 300，最后一窗完整输出。它需要上一窗预测后才能确定下一起点，单篇无法提前组成 Batch。固定窗口 B1 与固定窗口 BN 的对比用于分离纯 Batch 收益；逐窗基线与固定窗口的精度对比反映窗口策略差异。

速度测试在测试集中最长的 40 篇文章上运行，所有方法使用相同正文，分别预热并重复 3 次，交替正反顺序，CUDA 计时前后同步。计时包括字符编码、Batch 构造、模型前向和结果拼接，不包括模型加载及文件读写。记录中位耗时、字符每秒、前向次数、峰值已分配显存，以及相对固定 B1 的标签一致率。半精度矩阵运算可能使接近决策边界的标签随 Batch 形状变化，应以实测一致率为准。

## 原始证据

`runs/main/environment.json` 保存训练参数和依赖版本；`history.json` 保存每轮损失及验证指标；`best_epoch.json` 保存选模依据；`completion.json` 只有训练全部完成才生成。`runs/main/evaluation/` 保存整篇测试、吞吐量、显存和古文预测样例，不能用边界单元测试代替实验结果。

模型来源：[google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)。模型微调接口依据 [Hugging Face token classification 文档](https://huggingface.co/docs/transformers/tasks/token_classification)。人民日报语料由用户提供，不以下载来源冒充其原始来源；6 个文件的 SHA-256 保存在数据统计中。
