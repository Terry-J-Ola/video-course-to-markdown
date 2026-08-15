# 配置说明

## 运行环境与凭证

- Python 3.10 或更高版本
- 可访问阿里云 DashScope 的网络环境
- 系统 `PATH` 中存在 FFmpeg，或安装 `imageio-ffmpeg`
- Python 依赖：`pillow`、`numpy`、`imageio-ffmpeg`、`dashscope`

环境文件来源的解析规则如下：

```text
1. Existing process environment variables take precedence
2. 提供 --env-file 时，只读取该文件；不查找项目或用户私有 .env
3. 若未提供 --env-file，依次查找 <skill-root>/.env 和 %LOCALAPPDATA%\Codex\video-course-to-markdown\.env
```

所有模型均来自阿里云百炼，但当前按模型族使用两个 Key：

- Qwen 视觉与文本：`DASHSCOPE_QWEN_API_KEY` → `DASHSCOPE_API_KEY`
- Paraformer 音频转写：`DASHSCOPE_ASR_API_KEY` → `DASHSCOPE_API_KEY`

箭头表示优先使用左侧专用 Key，缺失时才使用右侧共享 Key。因此当前可在 `.env` 中分别配置两个专用 Key；后续两个 Key 统一后，只配置 `DASHSCOPE_API_KEY` 即可，无需改命令或代码。真实运行必须解析到 Qwen Key；除非指定 `--skip-asr`，也必须解析到 ASR Key。`--skip-asr` 时不要求 ASR Key。

可使用 `--env-file "D:\secure\video-course.env"` 指定用户私有环境文件。显式文件缺少当前运行所需的专用 Key 和共享回退 Key 时，配置仍然缺失，真实运行会失败，不会再查找项目或用户私有 `.env`。运行前仅检查 Key 是否已配置，绝不显示其值。任何 API Key 都不得进入 Markdown、JSON、日志或 Skill 包。

将依赖安装到隔离目录：

```powershell
python scripts/bootstrap_dependencies.py
```

## 模型配置

| 任务 | 默认模型 | 覆盖参数 | 说明 |
|---|---|---|---|
| 画面文字和视觉理解 | `qwen3-vl-flash` | `--visual-model` | 接收按组发送的图片 |
| 带时间戳音频转写 | `paraformer-v2` | `--asr-model` | 接收提取后的 16 kHz 单声道 MP3 |
| 学员稿转换与缺失修复 | `qwen-plus` | `--text-model` | 使用兼容 OpenAI 格式的文本接口 |

只有在确认账号具有访问权限后，才更换模型。画面模型不作为音频事实来源，音频必须独立转写。

## 常用参数

| 参数 | 作用 |
|---|---|
| `--mode auto|slides|live` | 选择抽帧策略 |
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

## 输出文件

统一入口生成关键帧总览、技术证据稿及其 assets、画面与音频证据 JSON、提取音频、业务讲义、内容保留校验 JSON、处理报告、`视频处理统计.csv`，以及可断点续跑的 `_video_course_work/` 中间目录。学员业务稿通过复核前，不要删除中间证据。

`视频处理统计.csv` 使用 UTF-8 BOM，可直接用 Excel 打开。同一输出目录下每个视频占一行；再次处理相同绝对路径的视频时更新其最新记录。表格包含开始/结束时间、总耗时、各阶段耗时、三个实际模型、视觉与文本输入/输出/总 Token、`Qwen总Token`，以及 `ASR音频时长（秒）`。Paraformer-v2 按音频时长而非 Token 计费，因此不伪造 ASR Token；断点续跑复用的旧视觉结果不重复计入本次 Token。成功视频记录完整用量，批量失败视频记录失败状态与错误信息；`--dry-run` 不创建文件。

输入为文件夹路径时，统一入口默认按相对路径排序并顺序处理第一层受支持视频；增加 `--recursive` 后递归发现视频。每个视频使用独立输出子目录，根输出目录共享 `视频处理统计.csv` 并生成 `批量处理报告.json`。失败视频会写入统计表和自己的失败处理报告，后续视频继续处理；任一视频失败会令批量命令最终返回退出码 1。批量模式不接受 `--title`。

断点续跑检查点会记录输入文件身份、阶段模型及相关抽帧或视觉分组参数。只有指纹完全兼容时才自动复用；输入、模型、`--scan-fps`、`--mode`、`--group-size` 或 `--min-spacing` 改变时，相应阶段会重新计算。`--skip-asr` 状态不会被后续正常 ASR 运行复用。处理报告和技术证据稿使用实际复用或生成产物的模型信息，不把本次请求的不同模型写成旧产物的来源。
