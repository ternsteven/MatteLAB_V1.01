# MATTELAB（Gradio 应用）

> 这是一个基于 **BiRefNet** 的二次开发项目，提供开箱即用的前景/抠图/背景替换 Web 界面（Gradio）。
>
> * 默认将 **HuggingFace** 上的模型权重下载到仓库根目录的 `models_local/`（可更改）。
> * 支持 **CPU** 与 **GPU**（NVIDIA CUDA / AMD ROCm）环境。
> * 提供 **一键下载权重脚本**、`venv`/Conda 环境方案与 **Docker** 镜像方案。
>
> ⚠️ **合规与许可**：本项目的二次开发遵从 BiRefNet 原仓库的许可与权重使用限制（详见文末「许可与致谢」）。若你使用的权重或上游数据集标注为**非商用**或有其他限制，请严格遵守相应条款。

---

## 功能特性

* 🧩 一键脚本从 HuggingFace 拉取权重到 `./models_local/…`，便于离线复现与版本管理。
* 🖼️ Gradio 图形界面：拖拽/选择图片即可得到透明背景或替换背景的结果。
* ⚙️ 可选 CPU / GPU 运行；提供 Dockerfile，免去本机装依赖。
* 🧪 结构化目录、可重复的环境声明（`requirements.txt`、`environment.yml`、`Dockerfile`）。

---

## 一键整合包

* ⚙️由于之前我代码写的实在是很乱，在拆解功能当中，很多功能可能暂时还存在问题。
* 🧪为了让小白更方便地使用，提供了一键整合包
* 🧩百度网盘链接：链接:https://pan.baidu.com/s/1I6mZLNFEBbyxHj6hABIe5A 提取码:4e4q
* 🧩解压码：TSUNE

---


## 目录结构（建议）

```
.
├─ src/                      # 你的业务/模型代码（含 birefnet 包装、预处理/后处理等）
├─ scripts/
│  └─ download_weights.py    # 一键下载 HF 权重到 ./models_local/
├─ models_local/             # 本地权重目录（默认 .gitignore 忽略）
├─ app_gradio_new.py         # 常规启动（默认 127.0.0.1:7860）
├─ main.py                   # 自动选择可用端口 + 自动打开浏览器
├─ requirements.txt          # Python 依赖（见下文 CPU/GPU 安装说明）
├─ environment.yml           # Conda/Mamba 环境声明
├─ Dockerfile                # 容器化运行
├─ README.md                 # 本文件
└─ 启动_双击这里启动.bat       # （如在 Windows 上提供，可一键启动）
```

---

## 快速开始（本地运行）

> 需要 Python ≥ 3.10。建议使用 **虚拟环境** 隔离依赖。

### 方式 A：`venv` + `pip`（通用）

```bash
# 1) 创建并激活虚拟环境（Windows 改为 .\.venv\Scripts\activate）
python -m venv .venv
source .venv/bin/activate

# 2) 升级 pip 并安装通用依赖（不含 GPU 定制项）
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) 下载模型权重到 ./models_local/
python scripts/download_weights.py --repo-id <你的HF仓库ID> --revision main

# 4) 启动（二选一）
# 自动选择端口并尝试自动打开浏览器（推荐本地体验）
python main.py
# 或：固定监听 127.0.0.1:7860
python app_gradio_new.py
```

### 方式 B：Conda / Mamba

```bash
# 创建并激活环境
conda create -n birefnet-app python=3.10 -y  # 或 mamba create -n birefnet-app python=3.10 -y
conda activate birefnet-app

# 安装依赖并下载权重
pip install -r requirements.txt
python scripts/download_weights.py --repo-id <你的HF仓库ID> --revision main

# 运行
python main.py
```

> Windows 用户如仓库内提供 `启动_双击这里启动.bat`，也可以直接双击运行（需先安装好依赖）。

---

## GPU / CPU 安装指南

> `requirements.txt` 中固定了 `torch==2.5.1`。默认从 PyPI 安装的是 **CPU 版**。若需要 **GPU 版**，请根据你的平台使用官方索引安装相同版本的 GPU 轮子。

### NVIDIA CUDA（示例：CUDA 12.1）

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### AMD ROCm（示例：ROCm 6.1）

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.1
```

### CPU（无需独显）

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

> 如你把 `torch/torchvision` 保留在 `requirements.txt` 内，也可以在安装整包时带上 `--index-url <上面的地址>`，达到相同效果。

---

## 模型权重下载与管理

本项目默认将权重下载到 **仓库根目录**的 `models_local/` 下：

```bash
# 下载到 models_local/<org__name> 子目录
python scripts/download_weights.py --repo-id <你的HF仓库ID>

# 不创建子目录，直接下载到指定目录根
python scripts/download_weights.py --repo-id <你的HF仓库ID> --no-subdir

# 仅拉取特定后缀（默认已覆盖常见 *.safetensors/*.bin 等）
python scripts/download_weights.py --repo-id <你的HF仓库ID> --allow-patterns "*.safetensors" "*.bin"

# 拉取完整快照（包含仓库所有文件）
python scripts/download_weights.py --repo-id <你的HF仓库ID> --allow-patterns ""
```

> 访问私有模型：
>
> ```bash
> huggingface-cli login
> # 或设置环境变量
> export HF_TOKEN=hf_xxx
> ```

> **离线运行**：权重下载完成后，保持 `models_local/` 目录在本地即可离线使用。

---

## Docker 方式（零环境成本）

```bash
# 1) 构建镜像（在项目根目录）
docker build -t birefnet-app:latest .

# 2) 预下载权重到宿主机（推荐）
mkdir -p models_local
python scripts/download_weights.py --repo-id <你的HF仓库ID> --revision main

# 3) 运行并映射端口/权重目录
# CPU 示例
docker run --rm -it -p 7860:7860 \
  -e APP_MODE=auto \
  -v "$PWD/models_local:/app/models_local" \
  birefnet-app:latest

# NVIDIA GPU（需宿主安装驱动和 nvidia-container-toolkit）
docker run --rm -it --gpus all -p 7860:7860 \
  -e APP_MODE=auto \
  -v "$PWD/models_local:/app/models_local" \
  birefnet-app:latest
```

> 容器内默认 `APP_MODE=auto`，即执行 `python main.py`；如需固定入口，可 `-e APP_MODE=direct` 以运行 `python app_gradio_new.py`。

---

## 使用方法

1. 打开浏览器访问控制台输出的本地 URL（例如 `http://127.0.0.1:7860` 或自动检测端口后的链接）。
2. 在 Gradio 界面上传图片，选择输出类型（透明 PNG / 替换背景等），点击开始。
3. 结果会在页面中展示，支持下载。

---

## 常见问题（FAQ）

* **OpenCV 报错 `libGL.so not found`（Linux）**：安装系统库 `libgl1`；Dockerfile 已包含。
* **MoviePy/FFmpeg 报错**：请安装 `ffmpeg`；Dockerfile 已包含。
* **私有模型下载 401**：`huggingface-cli login` 或设置 `HF_TOKEN` 后重试。
* **端口被占用**：使用 `main.py` 会在 `7860-7890` 范围内自动探测可用端口。
* **GPU 版 PyTorch 安装失败**：确认驱动/CUDA 版本匹配，使用上文的官方 `--index-url` 命令安装。

---

## 开发建议

* 将大文件（权重、数据、生成结果）放在 `models_local/`、`outputs/`、`runs/` 等目录，并在 `.gitignore` 中忽略，保持仓库精简。
* 对外发布时建议：

  * 在 GitHub 仓库仅包含代码与说明；
  * 权重托管在 HuggingFace（或 GitHub Releases 分卷）；
  * 使用同名 tag 对齐代码与权重版本。

---

## 许可与致谢

* **上游 BiRefNet（代码）**：请遵循 BiRefNet 原仓库的开源许可（MIT）。
* **模型权重**：若你使用的权重、数据集或第三方资源标注了**非商用**或其他限制（例如某些衍生/外部权重对商业用途有限制），在分发与使用时请严格遵守其条款。对于你自行训练/转换的权重，请在本仓库中明确注明对应许可与来源。
* **本项目代码**：请在此处补充你对本仓库的许可选择（例如 MIT / Apache-2.0 / AGPL-3.0 等），并在 `LICENSE` 文件中给出完整文本。
* **学术引用（BiRefNet）**：

  ```
  @article{zheng2024birefnet,
    title   = {Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
    author  = {Peng Zheng and Dehong Gao and Deng-Ping Fan and Li Liu and Jorma Laaksonen and Wanli Ouyang and Nicu Sebe},
    journal = {arXiv preprint arXiv:2401.03407},
    year    = {2024}
  }
  ```

> 若你的仓库公开发布，请在 README 顶部明确注明本项目基于 BiRefNet 的二次开发，并链接到上游仓库，以便他人了解来源与引用方式。

---

## 变更日志

* v1.0.1：初始公开；加入 `models_local/` 权重管理与 `scripts/download_weights.py`；补充 Docker 与 GPU 安装说明。
* v1.3：基本功能修复，代码结构化调整，新增半透明抠图功能，去白边功能。
* v1.4.1：新增涂抹抠图功能，指定区域抠图（代码调整中），上传一键整合包方便小白使用.

---

## 联系方式

* 问题或建议：欢迎提 Issue 或 PR。
* 商务/授权相关：请根据你所使用的权重与数据集条款，联系对应权利人或维护者获取授权。
