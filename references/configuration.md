# 配置说明

## 运行环境与凭证

- Python 3.10 或更高版本
- 可访问阿里云 DashScope 的网络环境
- 系统 `PATH` 中存在 FFmpeg，或安装 `imageio-ffmpeg`
- Python 依赖：`pillow`、`numpy`、`imageio-ffmpeg`、`certifi`、`openpyxl`
- DashScope HTTPS 请求使用显式 `certifi` 信任链并保持主机名与证书校验；不得通过关闭 SSL 校验规避连接错误。

环境文件来源的解析规则如下：

```text
1. Existing process environment variables take precedence
2. 提供 --env-file 时，只读取该文件；不查找项目或用户私有 .env
3. 若未提供 --env-file，依次查找 <skill-root>/.env 和 %LOCALAPPDATA%\Codex\video-course-to-markdown\.env
```

三个模型均来自阿里云百炼、共用一个 DashScope Key，并通过原生 REST 接口调用。推荐配置：

```dotenv
DASHSCOPE_API_KEY=sk-your-unified-api-key
```

也可在 CLI 使用 `--api-key "sk-..."`；它的优先级最高，并同时配置 qwen3-vl-flash、paraformer-v2 和 qwen-plus。批处理只通过继承环境把 Key 提供给子进程，不会把 `--api-key` 及其值拼进子命令。注意这种输入方式会留在 CMD 历史且可能短暂显示在系统进程参数中；共享机器优先使用私有 `.env` 或进程环境变量。

为兼容旧部署，Qwen 仍支持 `DASHSCOPE_QWEN_API_KEY` → `DASHSCOPE_API_KEY`，Paraformer 仍支持 `DASHSCOPE_ASR_API_KEY` → `DASHSCOPE_API_KEY`。统一 Key 用户无需配置这两个专用变量。真实运行必须解析到 Qwen Key；除非指定 `--skip-asr`，也必须解析到 ASR Key。`--skip-asr` 时不要求 ASR Key。

可使用 `--env-file "D:\secure\video-course.env"` 指定用户私有环境文件。显式文件缺少当前运行所需的专用 Key 和共享回退 Key 时，配置仍然缺失，真实运行会失败，不会再查找项目或用户私有 `.env`。运行前仅检查 Key 是否已配置，绝不显示其值。任何 API Key 都不得进入 Markdown、JSON、日志或 Skill 包。

将依赖安装到隔离目录：

```powershell
python scripts/bootstrap_dependencies.py
```

## 模型配置

| 任务 | 默认模型 | 覆盖参数 | 说明 |
|---|---|---|---|
| 画面文字和视觉理解 | `qwen3-vl-flash` | `--visual-model` | DashScope 原生 `multimodal-generation`，接收 Base64 图片组 |
| 带时间戳音频转写 | `paraformer-v2` | `--asr-model` | DashScope 原生异步 `audio/asr/transcription` |
| 学员稿转换与缺失修复 | `qwen-plus` | `--text-model` | DashScope 原生 `text-generation` |

原生端点分别为：

- 视觉：`/api/v1/services/aigc/multimodal-generation/generation`
- 文本：`/api/v1/services/aigc/text-generation/generation`
- ASR：`/api/v1/services/audio/asr/transcription`

实现直接使用经证书校验的 HTTPS，不依赖 DashScope Python SDK。原生响应在内部归一化后再写入证据、Token 统计和处理报告。

只有在确认账号具有访问权限后，才更换模型。画面模型不作为音频事实来源，音频必须独立转写。

## 常用参数

| 参数 | 作用 |
|---|---|
| `--mode auto|slides|live` | 选择抽帧策略 |
| `--api-key sk-...` | 为三个模型显式提供同一个 DashScope Key（会进入终端历史） |
| `--env-file file.env` | 指定私有环境文件 |
| `--visual-model name` | 指定视觉模型 |
| `--asr-model name` | 指定音频转写模型 |
| `--text-model name` | 指定业务稿文本模型 |
| `--scan-fps 2` | 设置高召回扫描频率 |
| `--group-size 4` | 设置每次 VLM 请求的帧数 |
| `--workers 3` | 设置并发 VLM 请求数 |
| `--min-spacing N` | 在 VLM 分析前按时间间隔减少关键帧 |
| `--corrections file.json` | 对画面文字和 ASR 应用人工复核修正 |
| `--skip-asr` | 处理没有有效音频的视频 |
| `--skip-business` | 只生成技术证据，不生成业务稿 |
| `--recursive` | 文件夹输入时同时处理所有子文件夹 |
| `--dry-run` | 只输出执行计划，不调用模型 |
| `--log-file path.jsonl` | 指定持久化 JSONL 日志路径 |
| `--log-max-mb 20` | 指定单个日志轮转阈值（MiB） |
| `--log-backups 5` | 指定轮转日志保留份数 |
| `--json-output` | 将终端切换为机器可读 JSON 事件，而不是默认短进度 |

## 输出文件

统一入口生成关键帧总览、技术证据稿及其 assets、画面与音频证据 JSON、提取音频、业务讲义、内容保留校验 JSON、处理报告、`视频处理统计.xlsx`、持久化 `视频处理日志.jsonl`，以及可断点续跑的 `_video_course_work/` 中间目录。批处理还会把每个成功视频的业务讲义复制到根输出目录的 `业务讲义汇总/`，各视频子目录中的原文件和其他产物保持不变。学员业务稿通过复核前，不要删除中间证据。

CLI 默认输出适合人工阅读的短进度，不显示绝对输入/输出路径和完整子命令。日志使用 UTF-8 JSON Lines，子进程每产生一行就立即追加并刷新；每条记录包含 `run_id` 和进程 ID。默认 20 MiB 轮转、保留 5 份，可用 `--log-max-mb` 和 `--log-backups` 修改。`--json-output` 模式下，输入扩展名等早期校验错误也保持单行 JSON。终端和日志统一脱敏当前配置的 Key；dry-run 不创建日志。

`视频处理统计.xlsx` 是原生 Excel 工作簿，不再生成 CSV，包含三个工作表：

- `逐视频统计`：同一输出目录下每个视频占一行；再次处理相同绝对路径的视频时更新其最新记录。记录开始/结束时间、总耗时、各阶段耗时、三个实际模型、视觉与文本输入/输出/总 Token、`单视频总Token`、`ASR音频时长（秒）`、状态及错误信息。
- `模型Token汇总`：按实际模型名称聚合视觉模型和文本模型的输入/输出/总 Token；Paraformer-v2 的 Token 单元格留空，并明确标注“不提供Token，按音频时长计费”。
- `批次汇总`：记录视频总数、成功/失败数、逐视频耗时之和、视觉总 Token、文本总 Token、`批次总Token` 和 ASR 音频总时长。

Paraformer-v2 按音频时长而非 Token 计费，因此不得伪造 ASR Token；`单视频总Token` 与 `批次总Token` 只汇总模型接口实际返回的 Token。断点续跑复用的旧视觉结果不重复计入本次 Token。成功视频记录完整用量，批量失败视频记录失败状态与错误信息；`--dry-run` 不创建工作簿。

输入为文件夹路径时，统一入口默认按相对路径排序并顺序处理第一层受支持视频；增加 `--recursive` 后递归发现视频。每个视频使用独立输出子目录，根输出目录共享 `视频处理统计.xlsx`、`业务讲义汇总/` 并生成 `批量处理报告.json`。汇总目录为扁平结构，文件名由可读相对路径、`_业务讲义_` 和 10 位稳定哈希组成，并限制到 220 个字符。`业务讲义汇总清单.json` 管理这些副本：后续批次会删除输入中已不存在的受管讲义，处理失败时保留上次成功副本，且不删除清单外的用户文件。`--skip-business` 不创建汇总目录，`--dry-run` 仅在报告中给出计划路径。普通失败视频会写入统计工作簿和自己的失败处理报告，随后继续处理；遇到 API Key 无效或模型权限不足时则停止后续模型调用，并把剩余视频标记为跳过。任一视频失败或成功视频的业务讲义复制失败，都会令批量命令最终返回退出码 1。批量模式不接受 `--title`。

## API 异常处理

- 缺少 Key：真实运行在创建批次输出目录和启动任何视频子进程之前失败；`--dry-run` 仍可在没有 Key 时检查计划。
- HTTP 401：归类为 `authentication`，表示 Key 无效或过期，不重试。
- DashScope 模型接口 HTTP 403：归类为 `permission`，表示当前账号没有模型或接口权限，不重试。OSS 上传或结果临时地址的 403 单独归类为 `resource_access`，不会误报为统一 Key 无权限。
- HTTP 408、429、5xx 以及连接超时、断线等网络错误：最多尝试 3 次。429 等响应带数值型 `Retry-After` 时优先按其等待，否则按 2 秒、4 秒指数退避；单次等待最多 60 秒。
- 无法解析或结构不完整的模型响应：最多尝试 3 次，只有通过结构校验的产物才会写入可复用检查点。

单视频失败时，CLI 返回退出码 1，并在该视频目录写入 `*_处理报告.json` 和统计失败行。失败报告记录 `error_category`、`http_status`、`provider_code`、`retryable`、`service`、阶段与耗时；已完成阶段的实际 Token 和 ASR 时长会尽可能保留。批处理会把这些字段传到 `批量处理报告.json`，而不是只记录笼统的子进程退出码。若错误类别为 `authentication` 或 `permission`，报告还会设置 `stopped_early: true`，记录 `stop_category`、`stop_reason` 和 `skipped_videos`。API Key 会在子进程输出、日志和报告写入前统一脱敏。

断点续跑检查点会记录输入文件身份、阶段模型、API 协议版本及相关抽帧或视觉分组参数。只有指纹完全兼容时才自动复用；输入、模型、协议、`--scan-fps`、`--mode`、`--group-size` 或 `--min-spacing` 改变时，相应阶段会重新计算。由旧 OpenAI 兼容视觉接口生成的检查点不会冒充原生接口产物。`--skip-asr` 状态不会被后续正常 ASR 运行复用。处理报告和技术证据稿使用实际复用或生成产物的模型信息，不把本次请求的不同模型写成旧产物的来源。
