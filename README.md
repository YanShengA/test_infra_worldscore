# Test Infra WorldScore

该目录是原有 infra 目录的重命名版本。脚本均以本目录为基准解析路径，可移动或改名。

## 快速开始

1) 确认配置文件中的输出路径和数据路径正确。

2) 运行 infra 脚本：

```bash
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /ML-vePFS/research_gen/tja/test_infra_worldscore/config.yaml
```

## 详细用法

### 1) 配置文件

主要配置文件（参考用，对于每个任务建议自己写一份）：
- `config.yaml` 用于默认流程
- `config_eval_only.yaml` 仅评测使用

关键字段说明：
- `run.name`: 运行名称，用于输出目录
- `run.output_root`: 日志和解析后的配置输出路径
- `run.mode`: `full` / `infer-only` / `eval-only`
- `env.worldscore_path`: 包含 `WorldScore/` 的工作区根目录
- `env.data_path`: WorldScore 数据集路径
- `wan.base_model_root`: 模型根目录
- `worldscore.model_name`: 对应 `WorldScore/config/model_configs` 的模型名
- `worldscore.runs_root_base`: 模型输出根目录
- `worldscore.output_dir`: 输出子目录（相对路径）
- `worldscore.sampled_json_path`: 采样数据集 json
- `compute.num_gpus`: 推理使用 GPU 数量（单机设置1-8，多节点全部设置为8）
- `compute.eval_num_jobs`: 评测并行进程数
- `compute.eval_auto_mean`: 分片评测后是否自动汇总

### 2) 运行模式

- 全流程：
```bash
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /ML-vePFS/research_gen/tja/test_infra_worldscore/config.yaml
```

- 仅评测：
```bash
bash /ML-vePFS/research_gen/tja/test_infra_worldscore/infra.sh \
  /ML-vePFS/research_gen/tja/test_infra_worldscore/config_eval_only.yaml
```

### 3) 日志位置

日志输出到：
- `<run.output_root>/logs/node_<rank>/infra.log`
- `<run.output_root>/logs/node_<rank>/infer.log`（运行推理时）
- `<run.output_root>/logs/node_<rank>/eval.log`（运行评测时）

示例：
- `/ML-vePFS/research_gen/tja/test_infra_worldscore/output/wan720720/logs/node_0/`

### 4) 常见问题

- 评测 checkpoint 缺失：
  如果看到 `worldscore/benchmark/metrics/checkpoints` 相关文件缺失报错，请确认 `WorldScore` 仓库中对应权重存在。

- 旧评测结果导致跳过：
  若已有 `evaluation.json` 导致评测被跳过，请删除后再运行评测。

- 多机评测离线报错：
  多节点时每台机器都需要 HF 缓存中存在 `bert-base-uncased`。建议将 HF 缓存同步到所有节点，或将 `HF_HOME` 指向共享存储。

### 5) 多机环境变量（可选）

如使用集群调度环境，可通过以下变量配置多机推理/评测：
- `MLP_WORKER_GPU`: 每节点 GPU 数量

脚本也支持标准 `torchrun` 变量：
- `MASTER_ADDR` / `MASTER_PORT`
- `NNODES` / `NODE_RANK` / `NPROC_PER_NODE`

## 目录结构

```
./
  infra.sh
  config.yaml
  config_eval_only.yaml
  tools/
    parse_config.py
    filter_cases.py
    run_infer.sh
    run_eval.sh
    evaluate_filtered.py
```



---


### 测试部分样本的部分指标 
1. 确认模型config 文件runs_root
.env 指定worldscore_output所在路径 MODEL_PATH
export $(grep -v '^#' .env | xargs)  

MODEL_PATH/
└── runs_root/
    └── worldscore_output/

2. 测部分的指标
## 命令
python /ML-vePFS/research_gen/tja/WorldScore/run_evaluate_parallel.py \
--model_name fantasy_world \
--myaspect_list "context_alignment,3d_consistency"
## 参数说明
--model_name 加载模型对应路径
--myaspect_list 要测试的指标，用逗号连接为字符串,必须包含在以下指标内
                如果不指定，默认为None,测试全部指标

## 输出
每个样本的evaluation.json



    aspect_list = {
    #         "static": [
    #             # Control
    #             "camera_control",
    #             "object_control",
    #             "content_alignment",
    #             # Quality
    #             "3d_consistency",
    #             "photometric_consistency",
    #             "style_consistency",
    #             "subjective_quality",
    #         ],
    #         # Dynamics
    #         "dynamic": [
    #             "motion_accuracy",
    #             "motion_magnitude",
    #             "motion_smoothness",
    #         ],
    #     }



### 测试一个样本的全部指标 

## 命令
python /ML-vePFS/research_gen/tja/WorldScore/run_evaluate_one_case.py
--model_name fantasy_world \
--video_dir "/ML-vePFS/research_gen/tja/worldscore_output/static/photorealistic/indoor/dining_spaces/049"


## 输出
每个样本的evaluation.json


