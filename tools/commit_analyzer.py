#!/usr/bin/env python3
"""vLLM 上游 commit 每日分析工具。

从 vllm-project/vllm 上游仓库获取提交历史，分类并高亮多模态相关变更，
通过阿里云百炼 API 生成中文摘要，并可选推送到钉钉文档。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
REMOTE_NAME = "vllm-oss"
DEFAULT_FIRST_DATE = "2025-12-01"
DINGTALK_FOLDER_URL = (
    "https://alidocs.dingtalk.com/i/nodes/"
    "lyQod3RxJKe9QjOMidvaye34Wkb4Mw9r"
)

# Tag regex: matches [Tag1][Tag2] or [Tag1/Tag2] at the start of subject
TAG_RE = re.compile(r"\[([^\]]+)\]")
PR_RE = re.compile(r"\(#(\d+)\)\s*$")

MULTIMODAL_TAGS = {
    "multimodal", "mm", "mm encoder", "vision", "vlm", "audio",
    "asr", "video",
}
MULTIMODAL_KEYWORDS = re.compile(
    r"multimodal|multi[-_]modal|vision[-_]language|vision|"
    r"\bimage\b|\bvideo\b|\baudio\b|\bvlm\b|\bmm_",
    re.IGNORECASE,
)
MULTIMODAL_PATHS = re.compile(
    r"vllm/multimodal/|tests/multimodal/",
)

CATEGORY_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"bugfix|bug\s*fix|fix\s*bug|crashfix", re.I), "Bug 修复"),
    (re.compile(r"feature|frontend|responsesapi", re.I), "新特性"),
    (re.compile(r"\bmodel[s]?\b", re.I), "模型支持"),
    (re.compile(r"perf|kernel|benchmark", re.I), "性能优化"),
    (re.compile(r"quantiz|compressed.tensors", re.I), "量化"),
    (re.compile(r"v1|core|attention|moe", re.I), "核心架构"),
    (re.compile(r"ci|build|cmake", re.I), "CI/构建"),
    (re.compile(r"rocm|xpu|cpu|ascend|intel.gpu|tpu", re.I), "硬件平台"),
    (re.compile(r"doc[s]?", re.I), "文档"),
    (re.compile(r"refactor|chore|clean|misc|minor|improvement", re.I),
     "重构/杂项"),
]

CATEGORY_ORDER = [
    "多模态",
    "Bug 修复",
    "新特性",
    "模型支持",
    "性能优化",
    "量化",
    "核心架构",
    "CI/构建",
    "硬件平台",
    "文档",
    "重构/杂项",
    "其他",
]


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)

    def load(self) -> dict:
        if not self.state_file.exists():
            return {"last_analyzed_date": None, "run_history": []}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("状态文件损坏，重置为首次运行: %s", e)
            return {"last_analyzed_date": None, "run_history": []}

    def save(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_date_range(
        self,
        args_start: str | None,
        args_end: str | None,
    ) -> tuple[str, str]:
        state = self.load()
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        if args_start:
            start = args_start
        elif state["last_analyzed_date"] is None:
            start = DEFAULT_FIRST_DATE
        else:
            # Next day after last analyzed
            last = date.fromisoformat(state["last_analyzed_date"])
            start = (last + timedelta(days=1)).isoformat()

        end = args_end if args_end else (
            today if state["last_analyzed_date"] is None else yesterday
        )
        return start, end

    def update(self, end_date: str, run_info: dict) -> None:
        state = self.load()
        state["last_analyzed_date"] = end_date
        state["run_history"].append(run_info)
        # Keep last 90 entries
        state["run_history"] = state["run_history"][-90:]
        self.save(state)


# ---------------------------------------------------------------------------
# GitManager
# ---------------------------------------------------------------------------

class GitManager:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", self.repo_path, *args],
            capture_output=True,
            text=True,
            check=check,
        )
        if result.returncode != 0 and check:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        return result.stdout.strip()

    def setup_remote(self) -> None:
        remotes = self._run("remote").splitlines()
        if REMOTE_NAME in remotes:
            self._run("remote", "set-url", REMOTE_NAME, UPSTREAM_URL)
            log.info("已更新 remote %s -> %s", REMOTE_NAME, UPSTREAM_URL)
        else:
            self._run("remote", "add", REMOTE_NAME, UPSTREAM_URL)
            log.info("已添加 remote %s -> %s", REMOTE_NAME, UPSTREAM_URL)

        log.info("正在 fetch %s main...", REMOTE_NAME)
        self._run("fetch", REMOTE_NAME, "main", "--no-tags")
        log.info("fetch 完成")

    def get_commits(self, start_date: str, end_date: str) -> list[dict]:
        # --after is exclusive, --before is inclusive; use day before start
        start_d = date.fromisoformat(start_date) - timedelta(days=1)
        # end_date should be inclusive, add one day for --before
        end_d = date.fromisoformat(end_date) + timedelta(days=1)

        fmt = "%H|%s|%ae|%ad|%an"
        output = self._run(
            "log", f"{REMOTE_NAME}/main",
            f"--after={start_d.isoformat()}",
            f"--before={end_d.isoformat()}",
            f"--format={fmt}",
            "--date=short",
        )
        if not output:
            return []

        commits = []
        for line in output.splitlines():
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue
            commits.append({
                "hash": parts[0],
                "subject": parts[1],
                "author_email": parts[2],
                "date": parts[3],
                "author_name": parts[4],
            })
        return commits

    def get_commit_files(self, commit_hash: str) -> list[str]:
        output = self._run(
            "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash,
            check=False,
        )
        return output.splitlines() if output else []


# ---------------------------------------------------------------------------
# CommitParser
# ---------------------------------------------------------------------------

class CommitParser:
    def __init__(self, git: GitManager):
        self.git = git

    def parse_tags(self, subject: str) -> list[str]:
        return TAG_RE.findall(subject)

    def parse_pr_number(self, subject: str) -> str | None:
        m = PR_RE.search(subject)
        return m.group(1) if m else None

    def categorize(self, tags: list[str]) -> str:
        tag_str = " ".join(tags)
        for pattern, category in CATEGORY_MAP:
            if pattern.search(tag_str):
                return category
        return "其他"

    def is_multimodal(
        self,
        subject: str,
        tags: list[str],
        changed_files: list[str] | None = None,
    ) -> bool:
        # Tag layer
        for tag in tags:
            if tag.lower().strip() in MULTIMODAL_TAGS:
                return True
            # Handle compound tags like "Multimodal][Core"
            for part in tag.split("/"):
                if part.strip().lower() in MULTIMODAL_TAGS:
                    return True

        # Keyword layer
        if MULTIMODAL_KEYWORDS.search(subject):
            return True

        # File path layer
        if changed_files:
            for f in changed_files:
                if MULTIMODAL_PATHS.search(f):
                    return True

        return False

    def process_commits(
        self,
        raw_commits: list[dict],
        check_files: bool = True,
    ) -> tuple[list[dict], dict[str, list[dict]]]:
        """Returns (multimodal_commits, other_commits_by_category)."""
        multimodal: list[dict] = []
        by_category: dict[str, list[dict]] = {}

        total = len(raw_commits)
        for i, commit in enumerate(raw_commits):
            if (i + 1) % 200 == 0:
                log.info("解析进度: %d/%d", i + 1, total)

            tags = self.parse_tags(commit["subject"])
            pr = self.parse_pr_number(commit["subject"])
            category = self.categorize(tags)

            # Check files only for non-obvious commits to save time
            changed_files = None
            if check_files and not self.is_multimodal(
                commit["subject"], tags
            ):
                changed_files = self.git.get_commit_files(commit["hash"])

            is_mm = self.is_multimodal(
                commit["subject"], tags, changed_files
            )

            enriched = {
                **commit,
                "tags": tags,
                "pr_number": pr,
                "category": category,
                "is_multimodal": is_mm,
            }

            if is_mm:
                multimodal.append(enriched)
            else:
                by_category.setdefault(category, []).append(enriched)

        return multimodal, by_category


# ---------------------------------------------------------------------------
# LLMSummarizer
# ---------------------------------------------------------------------------

class LLMSummarizer:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _call_api(self, messages: list[dict], max_retries: int = 3) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }).encode("utf-8")

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, OSError, KeyError) as e:
                wait = 2 ** attempt
                log.warning(
                    "LLM API 调用失败 (attempt %d/%d): %s, %ds 后重试",
                    attempt + 1, max_retries, e, wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
        return "（摘要生成失败，请查看 Commit 列表）"

    def _build_commit_text(self, commits: list[dict]) -> str:
        lines = []
        for c in commits:
            tags = "".join(f"[{t}]" for t in c["tags"]) if c["tags"] else ""
            pr = f" (#{c['pr_number']})" if c["pr_number"] else ""
            lines.append(f"- {c['date']} {tags} {c['subject']}{pr}")
        return "\n".join(lines)

    def _chunk_commits(
        self, commits: list[dict], chunk_size: int = 50
    ) -> list[list[dict]]:
        return [
            commits[i:i + chunk_size]
            for i in range(0, len(commits), chunk_size)
        ]

    def summarize_batch(
        self, commits: list[dict], context: str = "general"
    ) -> str:
        if not commits:
            return "该类别下无相关提交。"

        commit_text = self._build_commit_text(commits)

        if context == "multimodal":
            system_prompt = (
                "你是 vLLM 项目的技术分析师。请用中文为以下多模态相关的 "
                "git commits 生成简洁的技术摘要。重点说明：\n"
                "1. 新增或改进了哪些多模态能力（图像/视频/音频/VLM）\n"
                "2. 新增了哪些模型支持\n"
                "3. 对推理性能的影响\n"
                "4. 重要的 bug 修复\n"
                "分点列举主要变更，300-500字。"
            )
        else:
            system_prompt = (
                "你是 vLLM 项目的技术分析师。请用中文为以下 git commits "
                "生成简洁的技术摘要。分点列举主要变更，重点关注用户可见的"
                "功能变化和重要修复。200-400字。"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是 commit 列表：\n{commit_text}"},
        ]
        return self._call_api(messages)

    def _merge_summaries(self, summaries: list[str], context: str) -> str:
        if len(summaries) == 1:
            return summaries[0]

        combined = "\n\n---\n\n".join(
            f"第{i+1}批摘要：\n{s}" for i, s in enumerate(summaries)
        )
        messages = [
            {"role": "system", "content": (
                "你是 vLLM 项目的技术分析师。以下是分批生成的摘要，"
                "请将它们合并为一篇连贯的中文技术摘要，去除重复内容，"
                "保留所有关键信息。300-500字。"
            )},
            {"role": "user", "content": combined},
        ]
        return self._call_api(messages)

    def summarize_all(
        self,
        multimodal_commits: list[dict],
        other_commits_by_category: dict[str, list[dict]],
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        # Multimodal
        if multimodal_commits:
            log.info("正在生成多模态摘要 (%d commits)...", len(multimodal_commits))
            chunks = self._chunk_commits(multimodal_commits)
            batch_summaries = [
                self.summarize_batch(chunk, "multimodal") for chunk in chunks
            ]
            result["多模态"] = self._merge_summaries(
                batch_summaries, "multimodal"
            )

        # Other categories
        for category in CATEGORY_ORDER:
            if category == "多模态":
                continue
            commits = other_commits_by_category.get(category, [])
            if not commits:
                continue
            log.info(
                "正在生成 [%s] 摘要 (%d commits)...", category, len(commits)
            )
            chunks = self._chunk_commits(commits)
            batch_summaries = [
                self.summarize_batch(chunk, "general") for chunk in chunks
            ]
            result[category] = self._merge_summaries(
                batch_summaries, "general"
            )

        return result


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    @staticmethod
    def _format_commit_list(commits: list[dict]) -> str:
        lines = []
        for c in commits:
            short_hash = c["hash"][:9]
            # subject already contains [Tags] and (#PR), no need to duplicate
            lines.append(f"- `{short_hash}` {c['subject']}")
        return "\n".join(lines)

    def build(
        self,
        start_date: str,
        end_date: str,
        multimodal_commits: list[dict],
        other_commits_by_category: dict[str, list[dict]],
        summaries: dict[str, str],
    ) -> str:
        total = len(multimodal_commits) + sum(
            len(v) for v in other_commits_by_category.values()
        )
        mm_count = len(multimodal_commits)
        pct = f"{mm_count / total * 100:.1f}" if total > 0 else "0"

        # Title: use compact form for single-day
        if start_date == end_date:
            title_range = start_date
        else:
            title_range = f"{start_date} ~ {end_date}"

        parts = [
            f"# vLLM 上游提交分析报告",
            f"**时间范围：** {title_range}",
            f"**统计：** 总 {total} 条 | 多模态 {mm_count} 条（占比 {pct}%）",
            "",
            "---",
            "",
        ]

        # Multimodal section (always first)
        if multimodal_commits:
            parts.append("## 多模态相关变更（重点关注）")
            parts.append("")
            if "多模态" in summaries:
                parts.append("### AI 摘要")
                parts.append("")
                parts.append(summaries["多模态"])
                parts.append("")
            parts.append("### Commit 列表")
            parts.append("")
            parts.append(self._format_commit_list(multimodal_commits))
            parts.append("")
            parts.append("---")
            parts.append("")

        # Other categories
        for category in CATEGORY_ORDER:
            if category == "多模态":
                continue
            commits = other_commits_by_category.get(category, [])
            if not commits:
                continue
            parts.append(f"## {category}")
            parts.append("")
            if category in summaries:
                parts.append("### AI 摘要")
                parts.append("")
                parts.append(summaries[category])
                parts.append("")
            parts.append("### Commit 列表")
            parts.append("")
            parts.append(self._format_commit_list(commits))
            parts.append("")
            parts.append("---")
            parts.append("")

        parts.append(
            "*本报告由 commit_analyzer.py 自动生成 | AI 摘要由 Qwen 提供*"
        )
        return "\n".join(parts)

    def save(
        self,
        content: str,
        output_dir: str,
        start_date: str,
        end_date: str,
    ) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        s = start_date.replace("-", "")
        e = end_date.replace("-", "")
        filename = f"commit_analysis_{s}_{e}.md"
        path = out / filename
        path.write_text(content, encoding="utf-8")
        return str(path)


# ---------------------------------------------------------------------------
# DingTalkMCPClient
# ---------------------------------------------------------------------------

class DingTalkMCPClient:
    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self._initialized = False
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_jsonrpc(self, method: str, params: dict) -> dict:
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.mcp_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _initialize(self) -> None:
        if self._initialized:
            return
        result = self._send_jsonrpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "commit-analyzer", "version": "1.0.0"},
        })
        if "error" in result:
            raise RuntimeError(
                f"MCP initialize 失败: {result['error']}"
            )
        self._initialized = True
        log.info("钉钉 MCP 连接已初始化")

    def create_document(
        self,
        title: str,
        markdown: str,
        folder_url: str = DINGTALK_FOLDER_URL,
        max_retries: int = 2,
    ) -> dict:
        self._initialize()

        for attempt in range(max_retries):
            try:
                result = self._send_jsonrpc("tools/call", {
                    "name": "create_document",
                    "arguments": {
                        "name": title,
                        "markdown": markdown,
                        "folderId": folder_url,
                    },
                })
                if "error" in result:
                    raise RuntimeError(
                        f"MCP create_document 错误: {result['error']}"
                    )
                log.info("钉钉文档创建成功: %s", title)
                return result.get("result", {})
            except (urllib.error.URLError, OSError, RuntimeError) as e:
                wait = 2 ** attempt
                log.warning(
                    "钉钉 MCP 调用失败 (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        log.error("钉钉文档创建最终失败: %s", title)
        return {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM 上游 commit 分析工具",
    )
    parser.add_argument("--start-date", default=None,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output-dir", default="reports/",
                        help="报告输出目录")
    parser.add_argument("--state-file",
                        default="data/commit_analysis_state.json",
                        help="状态文件路径")
    parser.add_argument("--repo-path", default=".",
                        help="仓库路径")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 摘要")
    parser.add_argument("--no-dingtalk", action="store_true",
                        help="跳过钉钉推送")
    parser.add_argument("--no-file-check", action="store_true",
                        help="跳过 commit 文件路径检查（加速）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Strip empty strings from GitHub Actions inputs
    if args.start_date == "":
        args.start_date = None
    if args.end_date == "":
        args.end_date = None

    # State
    state_mgr = StateManager(args.state_file)
    start_date, end_date = state_mgr.get_date_range(
        args.start_date, args.end_date
    )
    log.info("分析时间范围: %s ~ %s", start_date, end_date)

    # Validate date range
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        log.info("起始日期晚于结束日期，无需分析")
        return

    # Git
    git = GitManager(args.repo_path)
    try:
        git.setup_remote()
    except subprocess.CalledProcessError as e:
        log.error("git 操作失败: %s\nstderr: %s", e, e.stderr)
        sys.exit(1)

    # Get commits
    log.info("正在获取 commits...")
    raw_commits = git.get_commits(start_date, end_date)
    log.info("获取到 %d 条 commits", len(raw_commits))

    if not raw_commits:
        log.info("该日期范围内无新 commit")
        # Still update state
        state_mgr.update(end_date, {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "commit_count": 0,
            "multimodal_count": 0,
            "report_file": None,
        })
        return

    # Parse & categorize
    parser = CommitParser(git)
    log.info("正在解析和分类 commits...")
    mm_commits, other_by_cat = parser.process_commits(
        raw_commits, check_files=not args.no_file_check
    )
    log.info(
        "多模态: %d 条, 其他: %d 条 (%d 个分类)",
        len(mm_commits),
        sum(len(v) for v in other_by_cat.values()),
        len(other_by_cat),
    )

    # LLM summarization
    summaries: dict[str, str] = {}
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if args.no_llm or not dashscope_key:
        if not args.no_llm and not dashscope_key:
            log.warning("DASHSCOPE_API_KEY 未设置，降级为 no-llm 模式")
    else:
        summarizer = LLMSummarizer(dashscope_key)
        summaries = summarizer.summarize_all(mm_commits, other_by_cat)

    # Build report
    builder = ReportBuilder()
    report = builder.build(
        start_date, end_date, mm_commits, other_by_cat, summaries
    )

    # Save locally
    report_path = builder.save(report, args.output_dir, start_date, end_date)
    log.info("报告已保存: %s", report_path)

    # Push to DingTalk
    mcp_url = os.environ.get("DINGTALK_MCP_URL", "")
    if args.no_dingtalk or not mcp_url:
        if not args.no_dingtalk and not mcp_url:
            log.warning("DINGTALK_MCP_URL 未设置，跳过钉钉推送")
    else:
        try:
            client = DingTalkMCPClient(mcp_url)
            # Title format per spec
            if start_date == end_date:
                doc_title = start_date
            else:
                doc_title = f"{start_date}~{end_date}"
            client.create_document(doc_title, report)
        except Exception as e:
            log.error("钉钉推送失败（不影响整体流程）: %s", e)

    # Update state
    state_mgr.update(end_date, {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "commit_count": len(raw_commits),
        "multimodal_count": len(mm_commits),
        "report_file": report_path,
    })
    log.info("状态已更新，分析完成")


if __name__ == "__main__":
    main()
