from pathlib import Path
from tclogger import OSEnver, logger

configs_root = Path(__file__).parents[1] / "configs"
envs_path = configs_root / "envs.json"
ENVS_ENVER = OSEnver(envs_path)
SEARCH_APP_ENVS = ENVS_ENVER["search_app"]
WEBSOCKET_APP_ENVS = ENVS_ENVER["websocket_app"]
LOGS_ENVS = ENVS_ENVER["logs"]

secrets_path = configs_root / "secrets.json"
secrets_template_path = configs_root / "secrets_template.json"

if secrets_path.exists():
    SECRETS = OSEnver(secrets_path)
else:
    SECRETS = OSEnver(secrets_template_path)
    logger.warn(
        f"WARN: secrets.json not found. Using secrets_template.json though. Please create {secrets_path}."
    )

BILI_DATA_ROOT = Path(SECRETS["bili_data_root"])
ELASTIC_ENVS = SECRETS["elastic"]
MONGO_ENVS = SECRETS["mongo"]
LLMS_ENVS = SECRETS["llms"]
PYRO_ENVS = SECRETS["pyro"]
MILVUS_ENVS = SECRETS["milvus"]
