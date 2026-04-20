# Team Development Guide

## 1. 代码协作

- 用 **GitHub** 管理代码。  
- `main` 分支只放相对稳定、可运行的版本。  
- 每个人在自己的功能分支中开发，不直接在 `main` 上长期修改。  
- 平时在 IDE 中正常编写和修改文件。  
- 一个小阶段完成后，及时进行 `add + commit`，形成清晰的版本记录。  
- 当当前阶段代码已经可运行、可共享或需要协同时，再 `push` 到远程仓库。  

### 协作要求

- 开始工作前，先拉取最新代码，避免长期基于旧版本开发。  
- 不要长时间只在本地积累大量改动而不提交、不同步。  
- 不要“憋大招”式开发，建议以**小步快跑**的方式推进：  
  - 完成一个小功能就提交一次  
  - 形成一个可测试版本就推送一次  
  - 接口、目录结构、依赖有变化时及时告知队友  
- 如果自己的改动会影响别人使用（如接口、配置项、运行方式变化），要同步更新说明文档。  

---

## 2. 分支

- **分支的作用**：解决“开发走哪条版本线”。  
- 每个人在自己的功能分支中进行开发，开发完成并测试后，再合并回总分支。  

### 分支使用原则

- 分支不是子文件夹，不是用来长期拆分模块存放代码的。  
- 项目的模块划分体现在目录结构中，例如 `vision/`、`embedded/`、`configs/`；分支体现的是不同任务或不同阶段的开发路线。  
- 建议分支命名尽量清晰，例如：  
  - `feature-vision`  
  - `feature-serial`  
  - `fix-config`  

### 注意事项

- 进入开发前先确认当前所在分支是否正确。  
- 不要在错误分支上写了一堆代码后才发现。  
- 已经被多人使用的分支，不要随意改名。  

---

## 3. 环境管理的关键思想

你们现在最需要守住的是以下几点。

### 3.1 代码、环境、配置分开管理

- **代码**：用 Git 管理  
- **环境**：用 Conda / `requirements` 文件管理  
- **配置**：用 `yaml` / `.env` 管理  

也就是说，代码文件、依赖环境、运行参数不要混在一起，更不要靠口头说明来传递。

### 3.2 训练环境和部署环境分开

- **训练环境**：给 Windows 本机或训练机使用，依赖较重，主要用于模型训练、实验、调试。  
- **部署环境**：给 Linux 主机使用，依赖尽量精简，主要用于最终运行和联调。  

不要默认认为“训练环境能跑”就等于“部署环境也能直接照搬”。训练环境和部署环境的目标不同，因此应分别维护。

### 3.3 环境靠文件重建，不靠口头复现

环境必须靠文件记录。主要依靠：

- `environment.yml`
- `requirements.txt`
- `README`

这样做的目的，是保证队友换一台电脑、后期部署到 Linux 主机时，仍然可以重新把环境搭起来。

### 3.4 新增依赖必须记录

如果某个人为了运行项目新增了依赖包，就必须及时记录到对应文件中，而不是只装在自己电脑上。  

例如：

- 新装了某个 Python 包  
- 修改了 Conda 环境  
- 增加了运行所需的系统依赖  

都应该：

1. 更新环境文件  
2. 在提交记录中说明  
3. 必要时提醒队友同步环境  

否则后面就很容易出现“我这里能跑，你那里跑不了”的情况。

### 3.5 配置不要写死在代码里

不要把下面这些内容直接写死在代码中：

- 本机磁盘路径  
- 个人用户名路径  
- `COM3` 这类串口号  
- Linux 下的设备路径  
- 只适用于某一台电脑的模型路径  

应该优先放到：

- 配置文件  
- `.env`  
- 相对路径  
- 命令行参数  

这样代码在 Windows 和 Linux 上都会更容易迁移。

### 3.6 及时同步，不要一股脑做

环境和配置管理里最重要的一点就是：**不要埋头做太久再一次性合并。**

建议在这些时机及时同步：

- 新增了关键依赖  
- 改了配置文件字段  
- 改了模块接口  
- 改了目录结构  
- 做出了一个可运行的小版本  

这样可以尽早发现环境不一致、接口不一致、配置不一致的问题，而不是到后期集中爆炸。

---

## 4. Windows 和 Linux 的推荐流程

### 推荐主方案

**Windows 主开发 / 主训练 + WSL/Linux 定期验证 + 最终部署到 Linux 主机**

### 理由

- 目前团队成员对 Windows 更熟悉，日常开发效率更高。  
- 本机 4060 可以承担开发和中小规模训练。  
- 最终又确实需要部署到 Linux 主机。  
- 因此最稳妥的方式是：**主开发放在 Windows，Linux 尽早介入验证，但不必一开始就全迁过去。**

### 这样做优于两种极端情况

- 不是一直只在 Windows 上开发，到最后才第一次上 Linux。  
- 也不是一开始就要求所有开发都在 Linux / WSL 中完成。  

更合理的做法是：

#### 平时

- 主要在 Windows 上进行代码编写、训练和快速调试。  

#### 阶段性检查

- 在 WSL 或 Linux 环境中定期验证代码能否运行，检查：  
  - 路径是否跨平台  
  - 配置是否合理  
  - 推理脚本能否启动  
  - 依赖是否完整  
  - 代码是否存在只适用于 Windows 的写法  

#### 最终阶段

- 在 Linux 主机上整理并重建部署环境  
- 完成最终部署、推理运行和硬件联调  


---

## 5. 建议的项目目录结构

还没检查过，ai初步生成的，后期细化

```text
intel-ai-project/
├─ README.md
├─ .gitignore
├─ .env.example
├─ docs/
│  ├─ team_development_guide.md
│  ├─ environment_guide.md
│  └─ deployment_guide.md
├─ app/
│  ├─ main.py
│  ├─ pipeline.py
│  └─ controller.py
├─ vision/
│  ├─ train.py
│  ├─ infer.py
│  ├─ export_model.py
│  ├─ dataset.py
│  ├─ model_utils.py
│  └─ transforms.py
├─ embedded/
│  ├─ serial_comm.py
│  ├─ protocol.py
│  └─ sensor_reader.py
├─ configs/
│  ├─ train.yaml
│  ├─ dev.yaml
│  └─ deploy.yaml
├─ scripts/
│  ├─ setup_train_env_win.ps1
│  ├─ setup_dev_env_linux.sh
│  ├─ setup_deploy_env_linux.sh
│  ├─ run_train.sh
│  ├─ run_infer.sh
│  └─ run_app.sh
├─ environment/
│  ├─ environment.train.win.yml
│  ├─ environment.dev.linux.yml
│  └─ environment.deploy.linux.yml
├─ requirements/
│  ├─ train.txt
│  ├─ dev.txt
│  └─ deploy.txt
├─ tests/
│  ├─ test_infer.py
│  ├─ test_serial.py
│  └─ test_config.py
├─ models/
│  └─ .gitkeep
├─ logs/
│  └─ .gitkeep
└─ data/
   └─ .gitkeep