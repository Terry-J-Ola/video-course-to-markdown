import argparse
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

try:
    from .env_config import get_qwen_api_key
except ImportError:
    from env_config import get_qwen_api_key


INSTRUCTION = r"""
请把下面的“视频解析审计稿”改写成可以直接交给学员阅读的中文课程讲义。

这不是技术报告，也不是逐帧日志。必须遵守：

1. 删除所有技术性说明，包括模型名称、VLM、ASR、OCR、置信度、JSON、role、confirmed、inferred、uncertain、frame、解析方式等。
2. 不按每张图片或每几秒生成一个小节。根据课程主题和教学步骤合并内容，每份讲义控制在 4～7 个主要章节。
3. 时间只作为章节定位，格式使用“（约 01:37）”；不要显示毫秒。
4. 相同的页面标题、人物说明、按钮和重复讲解只保留一次。
5. 保留真正的课程信息：画面正文、教学示例、问题与反馈、讲解要点、操作步骤、注意事项。
6. 画面中清晰可见的正文不能改写成“页面介绍了……”之类的摘要；示例句、定义和检查清单应保留原文。
   特别注意：弹窗、术语卡片、知识解释框即使文字很长，也属于课程正文，必须完整写入讲义；禁止只保留“点击查看解释”的提示而删掉解释内容。
7. 音频内容与画面内容融合成自然的讲义，不要出现“音频转写”“画面文字”这类数据来源标签。
8. 每个章节最多选择一张最有代表性的原图。图片链接必须从源稿中原样复制，不能编造或修改路径；没有合适图片可以不放。
9. 真人演示课程应写成清晰的服务步骤；知识讲解课程应按概念、方法、示例、注意事项组织。
10. 不补充源稿中不存在的业务知识。确有必要加入通用培训建议时，必须单独放在“补充建议”栏目，不得写成原视频结论。发现互相冲突或无法确认的内容时，以更保守的表达呈现。
11. 数字、数量、距离、顺序、条件、对比关系、术语释义和完整示范话术为高优先级信息，不得为了简洁而删减。
12. 在完成改写前，逐项核对输入末尾的“强制保留清单”；其中每一项都必须在最终讲义正文中出现。
13. 强制保留内容在正文中完整出现一次即可；“重点回顾”只概括要点，不要再次复制长段定义。

建议结构：

# 课程名称

## 学习目标

用 2～4 条说明学完后能够掌握什么。

## 一、主题名称（约 MM:SS）

用自然段、短列表、示例引用等方式呈现课程内容。

## 重点回顾

汇总 4～8 条最重要的可执行要点。

只输出最终 Markdown，不要解释改写过程，不要使用 Markdown 代码围栏。
""".strip()


FIELD_RE = re.compile(r"^>\s*-\s*`(?P<tag>[^`]+)`\s*(?P<text>.+?)\s*$", re.MULTILINE)
MANDATORY_CUES = re.compile(
    r"(?:所谓|俗称|是指|定义|区别|步骤|注意|必须|不得|不能|不超过|至少|至多|"
    r"准备|取衣|引导|跟随|确认|介绍|提醒|询问|整理|帮助|多准备|"
    r"\d+\s*[～~-]\s*\d+|\d+个|\d+点)"
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TECHNICAL_TERM_RE = re.compile(
    r"(?:\bVLM\b|\bASR\b|\bOCR\b|uncertain|confirmed|inferred|frame_\d+|\.json\b)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).lower()


def clean_extracted_text(text: str) -> str:
    text = re.sub(
        r"；(?:依据用户提供的原始画面|位于右侧|位于左侧|经原始画面复核|复核校正).*$",
        "",
        text,
    )
    return text.strip()


def extract_mandatory_blocks(source: str) -> list[str]:
    """Extract long/definition-like screen text that business rewriting may not drop."""
    blocks: list[str] = []
    seen: set[str] = set()
    for match in FIELD_RE.finditer(source):
        tag = match.group("tag").strip().lower()
        text = clean_extracted_text(match.group("text").strip())
        if "uncertain" in text.lower() or len(normalize_text(text)) < 8:
            continue
        normalized = normalize_text(text)
        keep = (
            tag == "popup"
            or (tag == "body" and (len(normalized) >= 20 or MANDATORY_CUES.search(text)))
            or (tag == "caption" and (len(normalized) >= 18 or MANDATORY_CUES.search(text)))
            or (tag == "other" and MANDATORY_CUES.search(text))
        )
        if not keep or normalized in seen:
            continue
        seen.add(normalized)
        blocks.append(text)
    return blocks


def block_is_present(block: str, content: str) -> bool:
    required = normalize_text(block)
    candidate = normalize_text(content)
    if required in candidate:
        return True
    if len(required) >= 65:
        ngram_size, threshold = 4, 0.82
    elif len(required) >= 20:
        ngram_size, threshold = 3, 0.30
    else:
        ngram_size, threshold = 2, 0.50
    if len(required) >= ngram_size:
        required_ngrams = {
            required[index : index + ngram_size]
            for index in range(len(required) - ngram_size + 1)
        }
        candidate_ngrams = {
            candidate[index : index + ngram_size]
            for index in range(len(candidate) - ngram_size + 1)
        }
        if required_ngrams:
            ngram_recall = len(required_ngrams & candidate_ngrams) / len(required_ngrams)
            if ngram_recall >= threshold:
                return True
    clauses = [
        normalize_text(part)
        for part in re.split(r"[。！？；;]", block)
        if len(normalize_text(part)) >= 8
    ]
    if not clauses:
        return False
    matched = sum(1 for clause in clauses if clause in candidate)
    return matched / len(clauses) >= 0.9


def missing_blocks(blocks: list[str], content: str) -> list[str]:
    return [block for block in blocks if not block_is_present(block, content)]


def output_quality(content: str, output: pathlib.Path) -> dict:
    technical_leakage = sorted(set(TECHNICAL_TERM_RE.findall(content)))
    missing_images: list[str] = []
    for raw_link in IMAGE_RE.findall(content):
        link = raw_link.strip().strip("<>")
        if link.startswith(("http://", "https://", "data:")):
            continue
        image_path = pathlib.Path(link)
        if not image_path.is_absolute():
            image_path = output.parent / image_path
        if not image_path.exists():
            missing_images.append(link)
    return {
        "technical_leakage": technical_leakage,
        "missing_images": missing_images,
        "image_count": len(IMAGE_RE.findall(content)),
    }


def preservation_prompt(blocks: list[str]) -> str:
    if not blocks:
        return ""
    items = "\n\n".join(f"{index}. {block}" for index, block in enumerate(blocks, 1))
    return (
        "\n\n---\n\n# 强制保留清单\n\n"
        "以下内容来自画面中的定义、长正文或硬性规则。必须写入最终讲义正文，"
        "可调整排版，但不得用一句摘要替代，也不得只写‘点击查看’：\n\n"
        + items
    )


def call_model(api_key: str, model: str, prompt: str, temperature: float) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown") :].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].rstrip()
    return content, result.get("usage", {})


def repair_missing_blocks(
    api_key: str,
    model: str,
    content: str,
    missing: list[str],
) -> tuple[str, dict]:
    items = "\n\n".join(f"{index}. {block}" for index, block in enumerate(missing, 1))
    prompt = f"""
下面是一份已经生成的中文课程讲义，但完整性校验发现有课程原文缺失。

请把“缺失内容”放回最合适的章节。保持现有业务讲义风格和图片链接，不增加技术术语，不删除现有课程事实。
缺失内容必须完整出现，不能只写摘要或“点击查看”。只输出修复后的完整 Markdown。

# 缺失内容

{items}

# 当前讲义

{content}
""".strip()
    return call_model(api_key, model, prompt, 0.1)


def append_fallback(content: str, missing: list[str]) -> str:
    if not missing:
        return content
    section = ["## 原课件知识补充", ""]
    for block in missing:
        section.extend([f"> {block}", ""])
    addition = "\n".join(section).rstrip()
    marker = "\n## 重点回顾"
    if marker in content:
        return content.replace(marker, "\n\n" + addition + marker, 1)
    return content.rstrip() + "\n\n" + addition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--audit-output")
    parser.add_argument("--max-repair-passes", type=int, default=2)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    source = pathlib.Path(args.source).read_text(encoding="utf-8")
    mandatory = extract_mandatory_blocks(source)
    output = pathlib.Path(args.output)
    audit_path = pathlib.Path(args.audit_output) if args.audit_output else output.with_suffix(".preservation.json")

    if args.validate_existing:
        content = output.read_text(encoding="utf-8")
        missing = missing_blocks(mandatory, content)
        quality = output_quality(content, output)
        audit = {
            "source": str(pathlib.Path(args.source)),
            "output": str(output),
            "source_chars": len(source),
            "output_chars": len(content),
            "mandatory_blocks": len(mandatory),
            "preserved_blocks": len(mandatory) - len(missing),
            "missing_blocks": missing,
            "mode": "validate-existing",
            **quality,
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False))
        return 1 if missing or quality["technical_leakage"] or quality["missing_images"] else 0

    api_key = get_qwen_api_key()
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_QWEN_API_KEY or DASHSCOPE_API_KEY is not configured"
        )

    prompt = INSTRUCTION + preservation_prompt(mandatory) + "\n\n---\n\n" + source
    content, initial_usage = call_model(api_key, args.model, prompt, 0.2)

    repair_usage: list[dict] = []
    missing = missing_blocks(mandatory, content)
    for _ in range(max(0, args.max_repair_passes)):
        if not missing:
            break
        content, usage = repair_missing_blocks(api_key, args.model, content, missing)
        repair_usage.append(usage)
        missing = missing_blocks(mandatory, content)

    fallback_appended = len(missing)
    if missing:
        content = append_fallback(content, missing)
        missing = missing_blocks(mandatory, content)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    quality = output_quality(content, output)
    audit = {
        "source": str(pathlib.Path(args.source)),
        "output": str(output),
        "source_chars": len(source),
        "output_chars": len(content),
        "mandatory_blocks": len(mandatory),
        "preserved_blocks": len(mandatory) - len(missing),
        "missing_blocks": missing,
        "repair_passes": len(repair_usage),
        "fallback_appended": fallback_appended,
        **quality,
        "usage": {"initial": initial_usage, "repairs": repair_usage},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
