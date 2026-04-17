from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess

from pathlib import Path
from urllib.parse import urlparse
from tclogger import logger


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TRACKED_CONFIG_FILES = {
    "configs/__init__.py",
    "configs/envs.json",
    "configs/envs.py",
    "configs/secrets.json.example",
}
TEXT_SUFFIXES = {
    "",
    ".cer",
    ".cfg",
    ".crt",
    ".env",
    ".gitignore",
    ".ini",
    ".json",
    ".key",
    ".md",
    ".pem",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
HIGH_CONFIDENCE_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----"),
    "openai-key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{16,}"),
    "stripe-live-key": re.compile(r"(?<![A-Za-z0-9])sk_live_[A-Za-z0-9]{16,}"),
    "github-token": re.compile(
        r"(?<![A-Za-z0-9])(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "gitlab-token": re.compile(r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9\-_]{20,}"),
    "google-api-key": re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z\-_]{20,}"),
    "aws-access-key": re.compile(r"(?<![A-Za-z0-9])(AKIA|ASIA)[0-9A-Z]{16}"),
    "slack-token": re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    "huggingface-token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    "jwt-token": re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}"
    ),
}
ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|token|secret|client[_-]?secret|password|passwd|authorization)\s*[\"']?\s*[:=]\s*[\"']([^\"'\n]{8,})[\"']"
)
URI_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:https?|mongodb|postgres(?:ql)?|mysql|redis|amqp)://[^\s\"'`<>()]+"
)
HOST_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[\"']?(?:host|hostname)[\"']?\s*[:=]\s*[\"']([^\"'\n]{2,})[\"']"
)

SAFE_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "host.docker.internal",
}
SAFE_SINGLE_LABEL_HOST_TOKENS = {
    "mock",
    "test",
    "example",
    "dummy",
    "fake",
    "primary",
    "secondary",
}
SAFE_DOMAIN_SUFFIXES = (
    ".example",
    ".example.com",
    ".test",
    ".invalid",
)


def is_placeholder_literal(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if not stripped:
        return True
    if stripped.startswith("${{") or stripped.startswith("$"):
        return True
    if "<" in stripped and ">" in stripped:
        return True
    if re.search(r"\bYOUR_[A-Z0-9_]+\b", stripped):
        return True
    if lowered.startswith(("test-", "dummy-", "fake-", "mock-", "example-")):
        return True
    if lowered.startswith(("your-", "your_", "set-", "replace-")):
        return True
    if lowered in {"test-key", "dummy-key", "fake-key", "mock-key"}:
        return True
    if "example" in lowered or "placeholder" in lowered or "replace-me" in lowered:
        return True
    return False


def tracked_files(root: Path = WORKSPACE_ROOT) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files = []
    for rel_path in result.stdout.split(b"\0"):
        if not rel_path:
            continue
        path = root / rel_path.decode("utf-8")
        if path.exists():
            files.append(path)
    return files


def staged_files(root: Path = WORKSPACE_ROOT) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / rel_path.decode("utf-8")
        for rel_path in result.stdout.split(b"\0")
        if rel_path
    ]


def should_scan(path: Path) -> bool:
    return (
        any(suffix in TEXT_SUFFIXES for suffix in path.suffixes)
        or path.name in TEXT_SUFFIXES
    )


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def read_staged_text_file(path: Path, root: Path = WORKSPACE_ROOT) -> str | None:
    rel_path = path.relative_to(root).as_posix()
    try:
        result = subprocess.run(
            ["git", "show", f":{rel_path}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None

    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_placeholder_secret(value: str) -> bool:
    stripped = value.strip()
    if is_placeholder_literal(stripped):
        return True
    if "*" in stripped:
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", stripped):
        return True
    if all(ch in "*._-" for ch in stripped):
        return True
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return True
    return False


def is_sensitive_host_literal(value: str) -> bool:
    raw_value = str(value or "").strip()
    if is_placeholder_literal(raw_value):
        return False

    candidate = raw_value if "://" in raw_value else f"scheme://{raw_value}"
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in SAFE_HOSTS:
        return False
    if host.endswith(SAFE_DOMAIN_SUFFIXES):
        return False

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host.endswith((".local", ".lan", ".internal")):
            return True
        if "." not in host:
            return not any(token in host for token in SAFE_SINGLE_LABEL_HOST_TOKENS)
        return False

    if ip.is_loopback or ip.is_unspecified:
        return False
    return ip.is_private or ip.is_link_local


def find_forbidden_tracked_paths(tracked_relpaths: set[str]) -> list[str]:
    violations: list[str] = []
    for relpath in sorted(tracked_relpaths):
        if not relpath.startswith("configs/"):
            continue
        if relpath in ALLOWED_TRACKED_CONFIG_FILES:
            continue
        violations.append(f"{relpath} must remain untracked")
    return violations


def display_path(path: Path, root: Path = WORKSPACE_ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_text(path: Path, text: str, *, root: Path = WORKSPACE_ROOT) -> list[str]:
    violations: list[str] = []
    for label, pattern in HIGH_CONFIDENCE_PATTERNS.items():
        if pattern.search(text):
            violations.append(f"{display_path(path, root)}: matched {label}")

    for uri in URI_PATTERN.findall(text):
        if is_sensitive_host_literal(uri):
            violations.append(
                f"{display_path(path, root)}: contains internal service URI"
            )

    for match in HOST_ASSIGNMENT_PATTERN.finditer(text):
        if is_sensitive_host_literal(match.group(1)):
            violations.append(
                f"{display_path(path, root)}: contains internal hostname assignment"
            )

    for match in ASSIGNMENT_PATTERN.finditer(text):
        key_name = match.group(1)
        value = match.group(2)
        if not is_placeholder_secret(value):
            violations.append(
                f"{display_path(path, root)}: {key_name} appears to contain a live secret"
            )
    return list(dict.fromkeys(violations))


def scan_tracked_files(root: Path = WORKSPACE_ROOT) -> list[str]:
    violations: list[str] = []
    tracked = tracked_files(root)
    tracked_relpaths = {path.relative_to(root).as_posix() for path in tracked}
    violations.extend(find_forbidden_tracked_paths(tracked_relpaths))

    for path in tracked:
        if not should_scan(path):
            continue
        text = read_text_file(path)
        if text is None:
            continue
        violations.extend(scan_text(path, text, root=root))

    return violations


def scan_staged_files(root: Path = WORKSPACE_ROOT) -> list[str]:
    violations: list[str] = []
    staged = staged_files(root)
    staged_relpaths = {path.relative_to(root).as_posix() for path in staged}
    violations.extend(find_forbidden_tracked_paths(staged_relpaths))

    for path in staged:
        if not should_scan(path):
            continue
        text = read_staged_text_file(path, root)
        if text is None:
            continue
        violations.extend(scan_text(path, text, root=root))

    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan repository files for secrets")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Scan only staged files from the git index for faster pre-commit checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    violations = scan_staged_files() if args.fast else scan_tracked_files()
    mode = "fast staged" if args.fast else "full"
    if violations:
        logger.warn(f"Sensitive information scan failed ({mode} mode):")
        for violation in violations:
            logger.warn(f"- {violation}")
        return 1
    logger.okay(f"Sensitive information scan passed ({mode} mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
