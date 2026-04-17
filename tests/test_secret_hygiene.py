from pathlib import Path
from unittest.mock import MagicMock, patch

from debugs.scan_sensitive_info import find_forbidden_tracked_paths, scan_staged_files
from debugs.scan_sensitive_info import scan_text
from debugs.scan_sensitive_info import scan_tracked_files


def _join_parts(*parts: str) -> str:
    return "".join(parts)


def test_tracked_files_do_not_contain_live_secrets():
    violations = scan_tracked_files()
    assert violations == []


def test_sensitive_config_files_must_stay_untracked():
    violations = find_forbidden_tracked_paths(
        {
            "configs/envs.py",
            "configs/envs.json",
            "configs/secrets.json.example",
            "configs/secrets.json",
            "configs/cert.pem",
            "configs/elastic_ca_dev.crt",
        }
    )
    assert violations == [
        "configs/cert.pem must remain untracked",
        "configs/elastic_ca_dev.crt must remain untracked",
        "configs/secrets.json must remain untracked",
    ]


def test_fast_mode_scans_only_staged_index_content(tmp_path):
    staged_list = MagicMock(stdout=b"configs/secrets.json\0docker/.env.example\0")
    staged_secret = MagicMock(stdout=b'{"api_' + b'key": "real' + b'SecretToken123"}')
    staged_env = MagicMock(stdout=b"BILI_SEARCH_APP_PORT=21001\n")

    with patch(
        "debugs.scan_sensitive_info.subprocess.run",
        side_effect=[staged_list, staged_secret, staged_env],
    ):
        violations = scan_staged_files(root=tmp_path)

    assert violations == [
        "configs/secrets.json must remain untracked",
        "configs/secrets.json: api_key appears to contain a live secret",
    ]


def test_fast_mode_ignores_unstaged_worktree_content(tmp_path):
    secret_file = tmp_path / "configs" / "secrets.json.example"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    secret_file.write_text('{"api_key": "REAL_WORKTREE_SECRET"}', encoding="utf-8")

    staged_list = MagicMock(stdout=b"configs/secrets.json.example\0")
    staged_secret = MagicMock(stdout=b'{"api_key": "YOUR_OPENAI_API_KEY"}')

    with patch(
        "debugs.scan_sensitive_info.subprocess.run",
        side_effect=[staged_list, staged_secret],
    ):
        violations = scan_staged_files(root=tmp_path)

    assert violations == []


def test_scan_text_flags_internal_service_uris_and_hostnames():
    internal_url = _join_parts("http://", "ai122", ":21501/transcripts/BV1abc")
    private_ip_url = _join_parts("http://", "11.192.168.4", ":3000/v1/chat/completions")
    internal_host = _join_parts("xe", "on")
    host_assignment = _join_parts("host", ' = "', internal_host, '"')

    violations = scan_text(
        Path("docs/internal.md"),
        "\n".join(
            [
                f"bili_store: {internal_url}",
                host_assignment,
                f'proxy = "{private_ip_url}"',
            ]
        ),
        root=Path("."),
    )

    assert violations == [
        "docs/internal.md: contains internal service URI",
        "docs/internal.md: contains internal hostname assignment",
    ]


def test_scan_text_ignores_placeholders_loopback_and_mock_hosts():
    violations = scan_text(
        Path("configs/secrets.json.example"),
        "\n".join(
            [
                'endpoint = "http://YOUR_BILI_STORE_HOST:21501"',
                'base_url = "http://127.0.0.1:21001"',
                'mock_url = "http://mock-google:18100"',
                'host = "localhost"',
            ]
        ),
        root=Path("."),
    )

    assert violations == []


def test_scan_text_detects_additional_secret_formats():
    gitlab_token = _join_parts("glpat-", "abcdefghijklmnopqrstuvwxyz123456")
    aws_key = _join_parts("AKIA", "ABCDEFGHIJKLMNOP")
    jwt_token = ".".join(
        [
            _join_parts("eyJ", "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFsaWNlIiwiaWF0IjoxNTE2MjM5MDIyfQ",
            "c2lnbmF0dXJlLXZhbHVlLXNhbXBsZQ",
        ]
    )

    violations = scan_text(
        Path("notes.txt"),
        "\n".join(
            [
                f"token={gitlab_token}",
                f"aws={aws_key}",
                f"auth={jwt_token}",
            ]
        ),
        root=Path("."),
    )

    assert "notes.txt: matched gitlab-token" in violations
    assert "notes.txt: matched aws-access-key" in violations
    assert "notes.txt: matched jwt-token" in violations
