# 配置说明 · 数据集与输出路径

测试脚本用到的所有路径（数据集目录、输出目录、模型位置）都集中在配置文件里，
**改路径不用动代码**。每个人还能用一份「私人覆盖清单」写自己机器上的路径，
不进 git、互不冲突。

---

## 路径写在哪：三个层级

优先级从高到低 —— **命令行 > 私人清单 > 共享默认清单**。

| 层级 | 文件 / 方式 | 是否进 git | 什么时候用 |
|---|---|---|---|
| 共享默认 | [`vision/datasets.yaml`](vision/datasets.yaml) | ✅ 进 git | 团队通用默认值，一般不用动 |
| **私人覆盖（推荐）** | `vision/datasets.local.yaml` | ❌ 不进 git | **你自己机器上的常用路径，主要写这里** |
| 临时一次性 | 命令行参数 | — | 只这一次想换目录 / 只跑几张 |

> 私人清单只需写你要改的那几项，其余自动沿用共享默认值。
> 你写你的、队友写他的，`git pull/push` 不会因为路径冲突。

---

## 三步上手（写自己的路径）

1. 把 [`vision/datasets.local.yaml.example`](vision/datasets.local.yaml.example)
   复制一份，改名成 `vision/datasets.local.yaml`（去掉 `.example`）。
2. 在里面**只写你要改的项**，例如你只跑少量数据：
   ```yaml
   num_samples: 5
   image_dirs:
     - D:/你的少量图片目录
   ```
3. 直接跑脚本，会自动优先用你这份。队友在他机器上同理写他的。

不想建文件、只想临时改一次：
```bash
conda run -n intel python scripts/visualize_segmentation.py --image-dir D:/samples --num 5
conda run -n intel python scripts/visualize_segmentation.py --out-dir runs/my_test --gt-dir D:/gt
```

---

## 清单里能改哪些路径

（详见 [`vision/datasets.yaml`](vision/datasets.yaml) 里每一项的注释）

| 键名 | 含义 | 用到的脚本 |
|---|---|---|
| `image_dirs` | 要扫描图片的目录（可多个，递归找 jpg/png） | 可视化 |
| `num_samples` / `seed` | 随机采样张数 / 种子（跑少量就把数量改小） | 可视化 |
| `vis_output_dir` | 可视化结果输出目录 | 可视化 |
| `eval_datasets` | 评测用的各数据集：名字 + split 列表 + 图片目录 | 评测 |
| `eval_output_dir` | 评测结果输出目录 | 评测 |
| `gt_dir` | road 真值标注目录（算 IoU 才需要，没有就留空 `null`） | 可视化 |
| `model_xml` | 模型 IR 文件位置（一般不用动） | 两个都用 |

> 路径写法：相对路径以**项目根目录**为基准；也可以直接写绝对路径（如 `D:/...`、`E:/...`）。

---

## 相关脚本

| 脚本 | 作用 | 输出 |
|---|---|---|
| `scripts/visualize_segmentation.py` | 抽样出图，目检分割效果；有标注时算 road IoU | `vis_output_dir` |
| `scripts/segmentation_eval.py` | 批量跑数据集，统计延迟 / FPS | `eval_output_dir` |

**注意事项**
- 跑之前先进 `intel` 环境：`conda run -n intel python scripts/xxx.py`（分割依赖只装在这个环境）。
- 准确率（road IoU）需要数据集的标注 mask，没有标注只能目检。各数据集标注格式不同，
  对应读取逻辑在 `visualize_segmentation.py` 的 `find_gt_road_mask` 里，按实际格式调整。
