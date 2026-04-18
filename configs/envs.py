from pathlib import Path
from tclogger import OSEnver, logger

configs_root = Path(__file__).parents[1] / "configs"
envs_path = configs_root / "envs.json"
ENVS_ENVER = OSEnver(envs_path)
SEARCH_APP_ENVS = ENVS_ENVER["search_app"]
LOGS_ENVS = ENVS_ENVER["logs"]

secrets_path = configs_root / "secrets.json"
secrets_example_path = configs_root / "secrets.json.example"

if secrets_path.exists():
    SECRETS = OSEnver(secrets_path)
else:
    SECRETS = OSEnver(secrets_example_path)
    logger.warn(
        f"WARN: secrets.json not found. Using secrets.json.example instead. Please create {secrets_path}."
    )

BILI_DATA_ROOT = Path(SECRETS["bili_data_root"])
ELASTIC_PRO_ENVS = SECRETS["elastic_pro"]
ELASTIC_DEV_ENVS = SECRETS["elastic_dev"]
MONGO_ENVS = SECRETS["mongo"]
LLMS_ENVS = SECRETS["llms"]
TEI_CLIENTS_ENVS = SECRETS["tei_clients"]
TEI_CLIENTS_ENDPOINTS = TEI_CLIENTS_ENVS.get("endpoints", [])
try:
    GOOGLE_HUB_ENVS = SECRETS["google_hub"]
except Exception:
    GOOGLE_HUB_ENVS = {}
LLM_CONFIG = SEARCH_APP_ENVS.get("llm_config", "deepseek")
