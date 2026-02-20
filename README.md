# 模型推理评测框架

本框架提供了完整的模型推理和评测流程，支持单机、多机环境，可选择全流程运行或仅推理/仅评测模式。

## 快速开始

### 前置准备


1. **配置文件准备**：根据任务需求编辑或创建配置文件（见下方配置说明）
可以直接复制一份/ML-vePFS/research_gen/tja/test_infra_worldscore/config.yaml文件进行修改
必改项目：
- `run.name`: 运行名称，建议使用有意义的名称（如实验编号）
- `run.output_root`: 输出根路径，确保有写权限
- `wan.checkpoint_path`: 模型权重路径，必须指向有效的模型文件
- `wan.resolution`: 推理分辨率，建议根据模型能力和资源调整
- `worldscore.resolution`: 评测分辨率，建议与推理分辨率一致
- `compute.eval_num_shards`: 测评的节点数量（例如32卡为四节点）

2. **运行脚本**：
   ```bash
   bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
     /ML-vePFS/research_gen/tja/test_infra_worldscore/config.yaml
   ```

## 配置文件详解

### 预置配置文件

- **`config.yaml`**: 完整流程配置（推理+评测）
- **`config_eval_only.yaml`**: 仅评测配置

建议为每个任务创建独立配置文件，避免相互干扰。

### 核心配置字段

#### 运行控制 (`run`)
| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 运行名称，用于生成输出目录 | `wan720720` |
| `output_root` | 日志和配置输出根路径 | `/path/to/output` |
| `mode` | 运行模式：`full`(全流程) / `infer-only`(仅推理) / `eval-only`(仅评测) | `full` |

#### 环境路径 (`env`)
| 字段 | 说明 | 示例 |
|------|------|------|
| `worldscore_path` | WorldScore 仓库根目录 | `/ML-vePFS/research_gen/tja/WorldScore` |
| `data_path` | 数据集路径 | `/ML-vePFS/research_gen/tja/data` |

#### 模型配置 (`wan`)
| 字段 | 说明 | 示例 |
|------|------|------|
| `base_model_root` | 模型权重根目录 | `/path/to/models` |

#### WorldScore 配置 (`worldscore`)
| 字段 | 说明 | 示例 |
|------|------|------|
| `model_name` | 模型配置名（对应 `WorldScore/config/model_configs/` 下的配置） | `fantasy_world` |
| `runs_root_base` | 推理输出根目录 | `/path/to/runs` |
| `output_dir` | 输出子目录（相对路径） | `worldscore_output` |
| `sampled_json_path` | 采样数据集 JSON 路径 | `/path/to/sampled.json` |

#### 计算资源 (`compute`)
| 字段 | 说明 | 推荐值 |
|------|------|--------|
| `num_gpus` | 推理使用的 GPU 数量 | 单机: 1-8<br>多机: 8 (每节点) |
| `eval_num_jobs` | 评测并行进程数 | 4-8 |
| `eval_auto_mean` | 评测后是否自动汇总结果 | `true` |

### 配置示例

```yaml
run:
  name: "exp_wan720"
  output_root: "/ML-vePFS/research_gen/tja/test_infra_worldscore/output/exp_wan720"
  mode: "full"

env:
  worldscore_path: "/ML-vePFS/research_gen/tja/WorldScore"
  data_path: "/ML-vePFS/research_gen/tja/data"

wan:
  base_model_root: "/ML-vePFS/research_gen/tja/models"

worldscore:
  model_name: "fantasy_world"
  runs_root_base: "/ML-vePFS/research_gen/tja/runs"
  output_dir: "worldscore_output"
  sampled_json_path: "/ML-vePFS/research_gen/tja/WorldScore/sampled_cases.json"

compute:
  num_gpus: 8
  eval_num_jobs: 8
  eval_auto_mean: true
```

## 运行模式说明

### 1. 全流程模式 (`mode: full`)

执行完整的推理和评测流程：

```bash
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /ML-vePFS/research_gen/tja/test_infra_worldscore/config.yaml
```

**流程**：
1. 解析配置文件
2. 过滤测试用例
3. **推理阶段**（diffsynth 环境）：生成视频
4. **多机同步**（如有）：等待所有节点推理完成
5. **评测阶段**（worldscore6 环境）：计算各项指标

### 2. 仅推理模式 (`mode: infer-only`)

只执行推理，不进行评测：

```bash
# 修改 config.yaml 中的 run.mode = "infer-only"
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /path/to/config_infer_only.yaml
```

**适用场景**：
- 批量生成视频
- 在推理和评测阶段使用不同配置

### 3. 仅评测模式 (`mode: eval-only`)

跳过推理，直接评测已有结果：

```bash
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /ML-vePFS/research_gen/tja/test_infra_worldscore/config_eval_only.yaml
```

**适用场景**：
- 重新评测已有推理结果
- 使用不同评测参数
- 推理和评测分离执行

**注意**：确保配置中的 `worldscore.runs_root_base` 和 `worldscore.output_dir` 指向已有的推理输出目录。

## 日志和输出

### 日志文件位置

所有日志输出到 `<run.output_root>/logs/node_<rank>/` 目录：

```
<run.output_root>/
├── logs/
│   └── node_0/              # 节点 0 的日志
│       ├── infra.log        # 主流程日志
│       ├── infer.log        # 推理日志（如有）
│       └── eval.log         # 评测日志（如有）
├── resolved_config.json     # 解析后的完整配置
├── run.env                  # 运行时环境变量
└── filtered_cases.json      # 过滤后的测试用例
```

**示例路径**：
```
/ML-vePFS/research_gen/tja/test_infra_worldscore/output/wan720720/logs/node_0/
```

### 推理输出

推理结果保存在 `<worldscore.runs_root_base>/<worldscore.output_dir>/` 目录：

```
<runs_root_base>/
└── <output_dir>/
    └── <scene_category>/
        └── <scene_type>/
            └── <case_id>/
                ├── video.mp4          # 生成的视频
                ├── metadata.json      # 元数据
                └── evaluation.json    # 评测结果（评测后生成）
```

### 评测结果

每个样本的评测结果保存在对应案例目录的 `evaluation.json` 文件中。

如果启用了 `compute.eval_auto_mean: true`，会在评测完成后自动生成汇总文件：
- `<output_dir>/mean_scores.json`：所有样本的平均分数

## 常见问题排查

### 1. CLIP 模型 SHA256 校验失败

**错误信息**：
```
RuntimeError: Model has been downloaded but the SHA256 checksum does not not match
```

**原因**：CLIP 库使用 `~/.cache/clip` 作为缓存目录，该路径在容器中不持久化。

**解决方案**：脚本已自动处理，在运行时会创建符号链接：
```bash
~/.cache/clip -> /ML-vePFS/research_gen/tja/cache/shared_clip_cache
```

确保 `/ML-vePFS/research_gen/tja/cache/shared_clip_cache/` 中包含：
- `ViT-L-14.pt`
- `ViT-B-32.pt`（如需要）

### 2. 评测 Checkpoint 缺失

**错误信息**：
```
FileNotFoundError: worldscore/benchmark/metrics/checkpoints/xxx.pth
```

**解决方案**：
1. 检查 `WorldScore` 仓库是否完整
2. 确认对应的评测模型权重已下载到 `WorldScore/worldscore/benchmark/metrics/checkpoints/`

### 3. 评测被跳过（已有结果）

**现象**：日志显示 "evaluation.json already exists, skipping..."

**原因**：目标目录已存在 `evaluation.json` 文件

**解决方案**：
```bash
# 删除旧的评测结果
find <runs_root_base>/<output_dir> -name "evaluation.json" -delete

# 或删除特定样本的结果
rm <runs_root_base>/<output_dir>/<scene>/<case>/evaluation.json
```

### 4. 多机环境 HF 缓存问题

**错误信息**：
```
OSError: Can't load tokenizer for 'bert-base-uncased'
```

**原因**：多节点评测时，某些节点缺少 HuggingFace 缓存

**解决方案**：
- 方案 1：确保所有节点的 `/ML-vePFS/research_gen/tja/cache/shared_hf_cache/` 包含必要模型
- 方案 2：使用共享存储挂载 `HF_HOME`

### 5. GPU 内存不足

**错误信息**：
```
CUDA out of memory
```

**解决方案**：
1. 减少 `compute.num_gpus`（如果是推理阶段）
2. 减少 `compute.eval_num_jobs`（如果是评测阶段）
3. 调整模型配置中的 batch size

### 6. 离线模式下载失败

**错误信息**：
```
Offline mode is enabled, but no cached version found
```

**原因**：设置了 `TRANSFORMERS_OFFLINE=1` 但缓存中缺少模型

**解决方案**：
1. 在有网络的环境先运行一次，下载所需模型到缓存
2. 或手动下载模型到对应的缓存目录

## 多机分布式运行

### 环境变量配置

脚本支持通过环境变量配置多机分布式推理和评测：

#### 标准 PyTorch 分布式变量
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 主节点端口
- `NNODES`: 总节点数
- `NODE_RANK`: 当前节点编号（0 开始）
- `NPROC_PER_NODE`: 每节点的进程/GPU 数量

#### 队列系统变量（自动映射）
如果使用集群调度系统，脚本会自动从以下变量映射：
- `MLP_WORKER_0_HOST` → `MASTER_ADDR`
- `MLP_WORKER_0_PORT` → `MASTER_PORT`
- `MLP_WORKER_NUM` → `NNODES`
- `MLP_ROLE_INDEX` → `NODE_RANK`
- `MLP_WORKER_GPU` → `NPROC_PER_NODE`



## 框架目录结构

```
test_infra_worldscore/
├── infra.sh                    # 主入口脚本
├── config.yaml                 # 全流程配置示例
├── config_eval_only.yaml       # 仅评测配置示例
├── README.md                   # 本文档
├── tools/                      # 工具脚本目录
│   ├── parse_config.py         # 配置解析工具
│   ├── filter_cases.py         # 测试用例过滤
│   ├── run_infer.sh            # 推理执行脚本
│   ├── run_eval.sh             # 评测执行脚本
│   └── evaluate_filtered.py    # 分片评测工具
└── output/                     # 运行输出（自动生成）
    └── <run_name>/
        ├── logs/               # 日志目录
        ├── resolved_config.json
        ├── run.env
        └── filtered_cases.json
```

## 环境和依赖

### Conda 环境

框架使用两个独立的 Conda 环境：

1. **worldscore6** (`/ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/worldscore6`)
   - 用于：配置解析、评测
   - 主要依赖：pyiqa, CLIP, bert-base-uncased 等

2. **diffsynth** (`/ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/diffsynth`)
   - 用于：模型推理
   - 主要依赖：diffusion models, torch 等

### 缓存目录

框架使用以下共享缓存目录（在 `infra.sh` 中配置）：

```bash
HF_HOME=/ML-vePFS/research_gen/tja/cache/shared_hf_cache
TORCH_HOME=/ML-vePFS/research_gen/tja/cache/shared_torch_cache
~/.cache/clip -> /ML-vePFS/research_gen/tja/cache/shared_clip_cache  # 符号链接
```

### 离线模式

默认启用离线模式，确保不会在运行时下载模型：
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```


