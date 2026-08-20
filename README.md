# video-course-to-markdown

将本地课程视频转换为可追溯的中文 Markdown 讲义，并输出关键帧、画面/音频证据、处理报告以及逐视频耗时与模型用量统计。

支持单个视频和文件夹批处理，支持递归扫描子文件夹及断点续跑。输入与输出均使用本地路径；关键帧、音频和文本会发送到阿里云百炼 DashScope 完成模型推理。

## 功能概览

- 支持 `.mp4`、`.mov`、`.mkv`、`.avi`、`.webm`、`.m4v`。
- 默认使用 `qwen3-vl-flash` 分析画面、`paraformer-v2` 转写音频、`qwen-plus` 整理业务讲义，三者统一通过 DashScope 原生 REST 接口调用。
- 文件夹模式逐个处理视频；普通单视频失败后继续，API Key 无效或模型权限不足时停止后续模型调用。
- 批处理把所有成功视频的业务讲义额外复制到统一的 `业务讲义汇总/`，原视频子目录及其他产物保持不变。
- CLI 默认显示简短中文进度，并把完整路径、命令、子进程输出、耗时和错误持久化到 JSONL 日志。
- 生成技术证据稿和学员业务讲义，尽量避免摘要过程静默丢失课程内容。
- 生成 `视频处理统计.xlsx`，记录逐视频耗时、视觉/文本 Token、批次总 Token 和 ASR 音频时长。
- 通过带来源指纹的检查点复用已完成阶段，支持在中断后使用相同命令继续运行。
- 三个模型共用一个 DashScope API Key，可从 CLI、环境变量或私有 `.env` 配置，不写入 Markdown、JSON、Excel 或处理日志。

## 运行要求

- Codex 桌面端、Codex CLI 或 IDE 扩展。
- Python 3.10 或更高版本。
- 能访问阿里云百炼 DashScope 的网络。
- 系统已安装 FFmpeg，或允许脚本安装隔离的 `imageio-ffmpeg`。
- 一个可调用 qwen3-vl-flash、paraformer-v2 和 qwen-plus 的阿里云百炼 DashScope API Key。

Python 依赖包括 `pillow`、`numpy`、`imageio-ffmpeg`、`certifi` 和 `openpyxl`。缺少依赖时，Skill 会按照自身说明把它们安装到用户目录下的隔离位置，不污染项目源码目录。

## 安装

OpenAI 的 [Skills 官方说明](https://developers.openai.com/codex/skills) 支持使用 `$skill-installer` 从其他代码仓库安装 Skill。在 Codex 中发送：

```text
请使用 $skill-installer 从 https://github.com/Terry-J-Ola/video-course-to-markdown 安装根目录中的 skill，安装名称为 video-course-to-markdown。
```

安装后可输入 `$video-course-to-markdown` 检查是否能选中该 Skill。Codex 通常会自动发现新安装的 Skill；如果没有显示，重启 Codex 后再试。

> 当前仓库适合通过 `$skill-installer` 做个人或本地安装。若以后希望通过通用插件目录面向更广泛用户分发，可再将它封装为 Codex Plugin。

## 第一次使用

### 1. 配置 API Key

推荐把私有配置保存到：

```text
%LOCALAPPDATA%\Codex\video-course-to-markdown\.env
```

文件内容为：

```dotenv
DASHSCOPE_API_KEY=填写统一Key
```

也可以把 `.env` 放在其他安全位置，并在运行时指定 `--env-file`；旧的 Qwen/ASR 专用 Key 变量仍兼容。CLI 还支持 `--api-key "你的统一Key"`，该值同时供三个模型使用，但会进入 CMD 历史且可能短暂出现在进程参数中，因此共享机器优先使用私有 `.env`。不要把真实 Key 提交到 Git 仓库或粘贴到对话中。

配置读取优先级如下：

1. 当前进程已有的环境变量。
2. 显式传入的 `--env-file`；使用后不再回退查找其他 `.env`。
3. Skill 根目录的 `.env`。
4. `%LOCALAPPDATA%\Codex\video-course-to-markdown\.env`。

### 2. 先执行 dry-run

dry-run 只检查输入、配置和执行计划，不调用模型，也不创建输出目录。在 Codex 中发送：

```text
使用 $video-course-to-markdown，对 "D:\courses\inputs" 做一次 dry-run，输出目录为 "D:\courses\outputs"，包含子文件夹。
```

确认计划中的输入路径、输出路径和模型无误后再进行真实处理。

### 3. 执行第一次真实处理

```text
使用 $video-course-to-markdown，处理文件夹 "D:\courses\inputs"，输出到 "D:\courses\outputs"，包含子文件夹。
```

第一次真实运行如果缺少 Python 依赖，Codex 会先按 Skill 说明安装隔离依赖。处理时间取决于视频长度、关键帧数量、网络和百炼接口限流情况。

## 日常使用

### 处理单个视频

```text
使用 $video-course-to-markdown，处理 "D:\courses\lesson-01.mp4"，输出到 "D:\courses\outputs"。
```

单视频产物会集中在：

```text
D:\courses\outputs\lesson-01\
```

### 批量处理文件夹

只处理文件夹第一层：

```text
使用 $video-course-to-markdown，处理文件夹 "D:\courses\inputs"，输出到 "D:\courses\outputs"。
```

同时包含子文件夹：

```text
使用 $video-course-to-markdown，处理文件夹 "D:\courses\inputs"，输出到 "D:\courses\outputs"，包含子文件夹。
```

批处理按照相对路径排序后顺序执行，并为每个视频建立独立输出子目录。每个原业务讲义继续保留，同时额外复制到输出根目录的 `业务讲义汇总/`。汇总文件名由可读相对路径和稳定短哈希组成，避免路径扁平化碰撞和超长文件名；`业务讲义汇总清单.json` 只管理工具生成的副本。输出根目录还共享一份 `视频处理统计.xlsx` 和 `批量处理报告.json`。

### 常见可选要求

可以直接在自然语言请求中说明，也可以让 Codex 映射为对应参数：

| 需求 | 参数 |
|---|---|
| 只查看计划，不调用模型 | `--dry-run` |
| 递归处理子文件夹 | `--recursive` |
| 只生成技术证据稿 | `--skip-business` |
| 确认视频没有有效音频 | `--skip-asr` |
| PPT/录屏课件 | `--mode slides` |
| 真人演示视频 | `--mode live` |
| 自动判断抽帧模式 | `--mode auto` |
| 遇到接口限流时降低并发 | `--workers 1` 或 `--workers 2` |
| 应用人工术语修正 | `--corrections "D:\secure\corrections.json"` |
| 使用指定私有配置 | `--env-file "D:\secure\video-course.env"` |
| CLI 输入统一 DashScope Key | `--api-key "sk-..."` |
| 指定持久化日志路径 | `--log-file "D:\logs\video-course.jsonl"` |
| 调整日志轮转 | `--log-max-mb 20 --log-backups 5` |
| 输出机器可读终端事件 | `--json-output` |

人工修正文件是 UTF-8 JSON，例如：

```json
{
  "莱塞尔": "莱赛尔",
  "实洗": "石洗"
}
```

### 直接使用命令行

通常建议由 Codex 调用 Skill。如果需要在仓库根目录直接运行，CMD 示例为：

```cmd
python scripts\run_video_course_pipeline.py "D:\courses\inputs" "D:\courses\outputs" --recursive
```

显式输入一个统一 Key（同供三个模型）：

```cmd
python scripts\run_video_course_pipeline.py "D:\courses\inputs" "D:\courses\outputs" --recursive --api-key "sk-..."
```

上述写法会留下 CMD 历史记录；长期使用仍建议配置私有 `.env`。

单视频并显式指定私有 `.env`：

```cmd
python scripts\run_video_course_pipeline.py "D:\courses\lesson-01.mp4" "D:\courses\outputs" --env-file "D:\secure\video-course.env"
```

若提示缺少依赖，先执行：

```cmd
python scripts\bootstrap_dependencies.py
```

脚本会返回隔离依赖目录。后续统一入口会自动发现默认目录；也可以通过 `--pydeps` 显式传入。

### 终端显示与持久化日志

默认 CLI 只显示短视频名或相对路径、批次序号、阶段中文名称、耗时和状态，例如：

```text
[批次] 共 2 个视频
[日志] 视频处理日志.jsonl
[1/2] 章节一\课程.mp4
  [开始] 画面分析
  [完成] 画面分析（12.40 秒）
[1/2] 完成（38.20 秒）
```

终端不会打印完整绝对路径、完整 Python 子命令或子脚本的大段输出。真实运行会把这些详细信息以 UTF-8 JSON Lines 追加保存到 `视频处理日志.jsonl`：

- 批处理：`<输出根目录>\视频处理日志.jsonl`
- 单视频：`<输出目录>\<视频标题>\视频处理日志.jsonl`
- 自定义位置：增加 `--log-file "D:\logs\video-course.jsonl"`

日志包含运行开始/结束、完整输入输出路径、阶段命令、实时子进程标准输出与错误输出、耗时、退出码和异常堆栈；每条记录带同一批次的 `run_id`。API Key 会在终端与日志中心化脱敏。默认单个日志达到 20 MiB 后轮转并保留 5 份，可用 `--log-max-mb`、`--log-backups` 调整。自动化程序可增加 `--json-output`，包括早期校验失败在内均返回 JSON。dry-run 不创建日志文件。

## 输出内容

每个成功处理的视频通常包含：

| 产物 | 用途 |
|---|---|
| `*_业务讲义.md` | 面向学员阅读的中文业务讲义 |
| `*_技术证据稿.md` | 带时间信息和画面证据的可追溯底稿 |
| `*_技术证据稿_assets/` | 技术证据稿引用的关键帧图片 |
| `*_关键帧总览.jpg` | 快速人工复核画面覆盖情况 |
| `*_画面证据.json` | 结构化视觉分析结果 |
| `*_音频证据.json` | 带时间戳的音频转写结果 |
| `*_提取音频.mp3` | 从视频提取的音频 |
| `*_内容保留校验.json` | 业务讲义的缺失内容、技术泄漏和缺图检查 |
| `*_处理报告.json` | 模型、阶段、时间、输出路径与用量摘要 |
| `视频处理统计.xlsx` | 逐视频、逐模型及批次统计 |
| `_video_course_work/` | 支持复核与断点续跑的中间结果和检查点 |
| `视频处理日志.jsonl` | 可持久化检索的完整运行、阶段输出和错误日志 |

批处理根目录还会生成：

- `业务讲义汇总/`：集中存放所有成功视频的业务讲义副本；原业务讲义不移动、不删除。
- `批量处理报告.json`：包含视频总数、成功/失败数量、汇总目录、逐视频讲义复制状态和失败项。

使用 `--skip-business` 时不会创建业务讲义汇总目录；使用 `--dry-run` 时只报告计划路径，不创建目录。

Excel 工作簿包含三个工作表：

- `逐视频统计`：每个视频的开始/结束时间、总耗时、各阶段耗时、实际模型、视觉与文本 Token、ASR 音频时长、状态和错误信息。
- `模型Token汇总`：按实际模型名称汇总输入、输出和总 Token。
- `批次汇总`：汇总视频数量、成功/失败数量、处理总耗时、批次总 Token 和 ASR 音频总时长。

`paraformer-v2` 不返回 Token，统计表会按音频时长记录其用量，不会伪造 ASR Token。

## Skill 内部处理流程

```mermaid
flowchart TD
    A[本地视频或视频文件夹] --> B[输入、输出、Key 与依赖检查]
    B --> C[自适应扫描与关键帧提取]
    C --> D[生成关键帧总览图]
    C --> E[qwen3-vl-flash 分组分析画面]
    B --> F[提取 16 kHz 单声道音频]
    F --> G[paraformer-v2 带时间戳转写]
    E --> H[合并画面与音频证据]
    G --> H
    H --> I[生成技术证据稿]
    I --> J[qwen-plus 生成业务讲义]
    J --> K[内容保留校验与自动修复]
    K --> L[写入报告与 XLSX 统计]
    L --> M[批处理复制业务讲义到统一汇总目录]
```

1. **发现输入并校验配置**：识别单文件或文件夹，筛选支持的视频扩展名，解析输出目录、模型参数、API Key 和隔离依赖。dry-run 在这里输出计划后停止。
2. **自适应抽帧**：先按扫描频率提取候选帧，再根据 `auto`、`slides` 或 `live` 模式筛选关键画面。课件模式倾向保留页面变化，真人演示模式会控制时间间隔和重复画面。
3. **画面理解**：通过 DashScope 原生 `multimodal-generation` 把 Base64 关键帧组发送给 `qwen3-vl-flash`，提取可见文字、界面状态、步骤、数字和其他画面事实。请求支持有限并发、重试和结构化 JSON 校验；无效响应会保留为诊断材料，但不会被写成可复用检查点。
4. **音频转写**：通过 FFmpeg 提取 16 kHz 单声道 MP3，再通过 DashScope 原生异步 ASR 接口调用 `paraformer-v2`，生成带时间戳的句子。画面模型不会被当作音频事实来源。
5. **证据合并**：按时间线组合视觉与音频结果，复制引用图片，生成技术证据稿和结构化证据 JSON。若提供人工修正表，会在这一阶段统一修正术语。
6. **业务讲义转换**：通过 DashScope 原生 `text-generation` 调用 `qwen-plus`，在保留定义、数字、完整步骤、案例和原始话术的前提下，把技术证据稿整理成更适合学习的中文讲义。
7. **内容保留校验**：检查必保内容缺失、技术标记泄漏和图片缺失。发现缺失时最多执行两轮自动修复；仍无法恢复的原课程内容会明确补入讲义，而不是静默省略。
8. **统计、日志、报告与讲义汇总**：写入阶段耗时、实际模型和接口返回的 Token，并把完整运行细节追加到 JSONL 日志。批量模式会把所有视频汇总到同一工作簿，并将成功视频的业务讲义原样复制到 `业务讲义汇总/`；复制失败会写入批量报告并让命令返回非零状态，但不会删除原视频目录中的任何产物。

## 断点续跑

处理被中断后，使用相同输入、相同输出目录和相同参数再次运行即可。Skill 会检查输入文件身份、模型以及抽帧/分组参数：

- 输入、模型、API 协议版本和参数指纹一致的关键帧、视觉分组结果及 ASR 转写会被复用。
- 输入文件、模型或相关参数改变时，只重新计算受影响的阶段。
- `--skip-asr` 产生的跳过状态不会阻止后续正常执行 ASR。
- 已复用的旧模型结果不会被误记为本次指定的新模型，也不会重复计算为本次 Token。
- 从旧 OpenAI 兼容视觉接口迁移后，旧视觉检查点会自动失效并使用原生接口重新计算一次。

在业务讲义和证据完成复核前，建议保留 `_video_course_work/`。确认不再需要断点续跑和问题诊断后，可以删除该中间目录；正式 Markdown、JSON、图片、音频和 Excel 产物不受影响。

## 安全与隐私

- 输入视频和最终产物路径在本地，但模型处理所需的关键帧、音频和文本会上传到阿里云百炼。
- API Key 不应进入仓库、对话、日志或任何输出产物。
- 仓库已忽略 `.env` 和常见开发过程目录；提交前仍应使用 `git status` 检查是否误加入私密文件。
- 不要通过关闭 SSL 证书验证来解决网络问题；脚本使用 `certifi` 信任链并保留主机名与证书校验。

## 常见问题

**安装后找不到 Skill**

重启 Codex，然后输入 `$video-course-to-markdown` 或通过 `/skills` 查找。

**提示未配置 Qwen 或 ASR Key**

检查 `.env` 文件名、位置和变量名。真实运行必须有 Qwen Key；只有明确使用 `--skip-asr` 时才不要求 ASR Key。

**提示缺少 FFmpeg、Pillow、OpenPyXL 等依赖**

让 Codex 按 Skill 说明执行依赖引导，或在仓库根目录运行 `python scripts\bootstrap_dependencies.py`。

**遇到 DashScope 限流或临时网络错误**

HTTP 408、429、5xx 和临时网络错误最多尝试 3 次；服务端提供 `Retry-After` 时会优先遵循。仍失败时降低 `--workers`，并用同一命令断点续跑。

**API Key 无效或没有模型权限**

HTTP 401/403 不会无意义地重试。单视频会写入 `*_处理报告.json` 后退出；批处理会停止后续模型调用，将剩余视频标记为跳过，并在 `批量处理报告.json` 中写入 `stopped_early`、错误类别和 HTTP 状态。若 Key 根本没有配置，真实批处理会在创建输出目录和启动第一个视频前失败。修正 Key 或开通模型权限后，使用原命令即可断点续跑。

**批处理返回非零退出码**

只要有一个视频失败，批处理最终就会返回失败状态。普通视频级错误不会影响后续视频；永久凭证错误会停止后续模型调用，避免对整个文件夹重复发送必然失败的请求。查看根目录的 `批量处理报告.json`、`视频处理统计.xlsx` 和失败视频的处理报告。

**能否在新项目中直接使用**

安装在个人 Skill 目录后可跨项目使用，不需要每新建一个项目就重新安装。只有更换电脑、用户环境或删除已安装 Skill 后才需要重新安装。

## 开发校验

在仓库根目录运行：

```cmd
python -m unittest discover -s tests -v
python -m compileall -q scripts
```

`tests/` 是发布前回归测试的一部分，建议保留在仓库中；安装后的 Skill 正常运行不依赖测试目录。发布前还应使用 Codex 自带的 Skill 校验器检查目录结构和 `SKILL.md` 元数据。
