---
name: video-course-to-markdown
description: 将用户指定的本地 MP4、MOV、MKV、AVI、WEBM 或 M4V 课程视频或包含这些视频的本地文件夹批量转换到用户指定的本地目录，生成技术证据稿、中文业务讲义、批次业务讲义汇总目录，以及包含逐视频耗时、各模型 Token、单视频总 Token 和批次总 Token 的 XLSX 统计工作簿。默认使用 qwen3-vl-flash、paraformer-v2 与 qwen-plus，统一通过阿里云百炼 DashScope 原生 REST 接口调用并共用一个 API Key；Key 可通过 --api-key、环境变量、--env-file 或私有 .env 配置，且绝不显示或写入产物。
---

# 视频课程转 Markdown

采用“证据优先、业务转换、完整性校验”的可恢复流程处理课程视频：

`视频 → 自适应关键帧 + 音频 → VLM + ASR → 技术证据稿 → 学员业务稿 → 内容保留校验`

## 准备环境

1. 以本 `SKILL.md` 所在目录为基准解析所有脚本路径。
2. 输入是用户指定的本地 MP4、MOV、MKV、AVI、WEBM 或 M4V 路径，或包含这些视频的文件夹路径；输出是用户指定的本地目录，并应为本次处理独立指定。
3. 配置模型、依赖、凭证或降级策略时，读取 `references/configuration.md`。
4. 三个模型共用一个 DashScope API Key，并全部通过 DashScope 原生 REST 接口调用：视觉使用 `multimodal-generation`、文本使用 `text-generation`、转写使用异步 `audio/asr/transcription`。CLI 显式提供 `--api-key` 时优先使用它，并仅通过当前进程环境传给各模型调用，不把它加入批处理子命令、日志或产物；否则依次读取现有 `DASHSCOPE_API_KEY`、显式 `--env-file`、项目 `.env` 和用户私有 `.env`。旧的 Qwen/ASR 专用变量仍兼容。直接在命令行提供 Key 会留在终端历史和短暂出现在进程参数中，因此无人值守或共享机器优先使用环境变量/私有 `.env`。
5. 关键帧、音频和文本会发送到阿里云百炼 DashScope 处理。
6. 仅在缺少依赖时安装隔离运行环境：

```powershell
python scripts/bootstrap_dependencies.py
```

将脚本返回的目录传给 `--pydeps`，或设置环境变量 `VIDEO_COURSE_PYDEPS`。

## 执行完整流程

优先运行统一入口：

```powershell
python scripts/run_video_course_pipeline.py `
  "D:\courses\service-training.mp4" `
  "D:\courses\service-training-output" `
  --mode auto
```

也可显式输入统一 Key：`--api-key "sk-..."`。该 Key 同时用于画面、音频转写和文本模型；不要把真实值写进脚本、仓库或对话。

使用私有环境文件时：

```powershell
python scripts/run_video_course_pipeline.py `
  "D:\courses\service-training.mp4" `
  "D:\courses\service-training-output" `
  --mode auto `
  --env-file "D:\secure\video-course.env"
```

已知是 PPT 或录屏课件时使用 `--mode slides`；已知是真人演示时使用 `--mode live`；无法判断时保留 `auto`。

## 批量处理文件夹

将文件夹作为第一个位置参数即可顺序处理其中所有受支持的视频：

```cmd
python scripts\run_video_course_pipeline.py "D:\courses\incoming" "D:\courses\batch-output"
```

默认只处理文件夹第一层；需要包含子文件夹时增加 `--recursive`。每个视频写入根输出目录下的独立子目录，原有业务讲义继续保留；同时把成功视频的业务讲义复制到根目录的 `业务讲义汇总` 文件夹。汇总文件名包含可读相对路径和稳定短哈希，避免扁平化碰撞并限制文件名长度；`业务讲义汇总清单.json` 记录受管副本，只清理已从输入批次删除的受管讲义，不触碰用户自行放入的文件。根目录还统一生成 `视频处理统计.xlsx`、`批量处理报告.json` 和 `视频处理日志.jsonl`。XLSX 包含 `逐视频统计`、`模型Token汇总`、`批次汇总` 三个工作表。普通的单视频失败会记录失败报告并继续下一个；若 DashScope 返回 API Key 无效或模型权限不足，则停止调用后续视频并在批量报告中标记为跳过，避免重复无效请求。视频处理或业务讲义汇总复制失败时，批次最终返回非零退出码。`--skip-business` 时不创建汇总目录，`--dry-run` 只报告计划路径而不创建目录。批量模式不使用 `--title`。

默认配置：

- 画面识别：`qwen3-vl-flash`
- 带时间戳音频转写：`paraformer-v2`
- 业务稿转换：`qwen-plus`
- 自适应扫描频率：每秒 2 帧
- 每次 VLM 请求：4 张关键帧

需要先检查执行计划时使用 `--dry-run`。只需要技术证据稿时使用 `--skip-business`。只有确认视频没有有效音频时才使用 `--skip-asr`。

CLI 默认只显示短视频名、批次进度、阶段中文名称、耗时和成功/失败状态，不在终端打印完整绝对路径、完整子命令或子脚本输出。子进程输出会实时流入 UTF-8 JSON Lines 日志，每条记录带同批次 `run_id`；默认日志达到 20 MiB 时轮转并保留 5 份。使用 `--log-file`、`--log-max-mb` 和 `--log-backups` 可调整；自动化程序使用 `--json-output`。所有终端与日志内容统一脱敏 API Key；`--dry-run` 不创建输出目录和日志文件。

## 交付前复核

读取 `references/workflow.md`，按照其中的证据优先级和质量标准复核。

必须完成：

1. 查看关键帧总览图，重点检查短时弹窗、悬停解释、题目反馈和文字密集页面。
2. 比较画面术语与 ASR 结果。专业术语冲突时，优先采用清晰可见且经过确认的画面文字。
3. 需要人工修正时，创建 UTF-8 JSON 修正表，例如：

```json
{
  "莱塞尔": "莱赛尔",
  "实洗": "石洗"
}
```

使用 `--corrections "<corrections.json>"` 重新执行。

4. 打开内容保留校验 JSON，确认：
   - `missing_blocks` 为空；
   - `technical_leakage` 为空；
   - `missing_images` 为空。
5. 不得仅因为文档读起来流畅就直接交付。定义、弹窗正文、数字、距离、顺序步骤、对比关系、完整案例和原始话术都属于必须保护的课程内容。
6. 将视频中没有明确表达的通用经验放入“补充建议（非视频原文）”，不得与原课程结论混写。

## 断点续跑与修复

在相同输出目录重新执行时，仅当输入文件身份、阶段模型、API 协议版本和相关抽帧/分组参数与检查点一致时，才复用已有扫描帧、VLM 分组结果和 ASR 转写；迁移前由 OpenAI 兼容接口生成的视觉检查点会自动重新计算。`--skip-asr` 生成的空结果只标记为跳过状态，之后取消该参数会正常执行转写。完成校验前保留 `_video_course_work`。

模型调用失败时：

- 真实运行会在创建批次输出和启动子进程前检查 Key 是否已配置；缺失时立即失败；
- HTTP 401（Key 无效/过期）和 403（模型权限不足）不重试；批处理停止后续模型调用，并将剩余视频标记为跳过；
- HTTP 408、429、5xx 和临时网络错误最多尝试 3 次，优先遵循服务端 `Retry-After`，否则指数退避；
- 遇到限流时降低 `--workers`；
- 更换模型名称前先验证当前账号的模型访问权限；
- 始终将画面解析和音频转写分开处理；
- 不得用没有证据的摘要代替缺失的画面正文。

单视频执行失败时也必须检查该视频的 `*_处理报告.json`；其中包含 `error_category`、HTTP 状态、服务端错误码、是否可重试、失败阶段和已产生的用量。批处理的 `批量处理报告.json` 会保留同一错误详情，并用 `stopped_early`、`stop_category` 和 `skipped_videos` 说明是否因永久凭证错误提前停止。任何报告或日志都不得包含 API Key。

如果校验发现必保内容缺失，重新运行学员稿转换脚本。脚本最多执行两轮自动修复；仍有缺失时，将原课程内容明确补入“原课件知识补充”，避免静默丢失。

## 交付产物

主要交付学员业务讲义；批处理时可直接交付根输出目录下的 `业务讲义汇总`，同时保留各视频子目录中的原始业务讲义和其他证据。根据需要同时提供技术证据稿、内容保留校验 JSON、关键帧总览图、处理报告和 `视频处理日志.jsonl`。每次完整处理成功后，检查或交付 `视频处理统计.xlsx`：批处理模式下在根输出目录，单视频模式下在该视频的输出子目录（`<输出目录>\<标题>\`）里。`逐视频统计` 按视频路径保留最新一行，记录总耗时、各阶段耗时、实际模型、视觉与文本输入/输出/总 Token、`单视频总Token` 和 Paraformer 音频时长；`模型Token汇总` 按实际模型名称聚合；`批次汇总` 记录视频数量、成功/失败数、视频处理总耗时和 `批次总Token`。Paraformer 不返回 Token，必须明确标注为按音频时长计费，不得伪造 Token。批量处理时还要检查 `批量处理报告.json` 中的视频失败项及 `business_lecture_failures`；即使学员讲义是主要产物，也要保留可供追溯的技术证据。
