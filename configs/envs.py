from pathlib import Path

from tclogger import OSEnver, logger

configs_root = Path(__file__).parents[1] / "configs"
envs_path = configs_root / "envs.json"
ENVS_ENVER = OSEnver(envs_path)

secrets_path = configs_root / "secrets.json"
secrets_template_path = configs_root / "secrets_template.json"

if secrets_path.exists():
    SECRETS = OSEnver(secrets_path)
else:
    SECRETS = OSEnver(secrets_template_path)
    logger.warn(
        f"WARN: secrets.json not found. Using secrets_template.json though. Please create {secrets_path}."
    )

VIDEOS_ROOT = SECRETS["videos_root"]
