from pathlib import Path
from unittest.mock import MagicMock, patch

from debugs.scan_sensitive_info import find_forbidden_tracked_paths, scan_staged_files
from debugs.scan_sensitive_info import scan_tracked_files


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
    staged_env = MagicMock(stdout=b"BILI_SEARCH_APP_PORT=21031\n")

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
