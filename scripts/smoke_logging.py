import os
import logging
from doc_studio.core.config import Settings
from doc_studio.core.logging import setup_logging


os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "test_key")
cfg = Settings.load()

setup_logging(cfg.log_level)
log = logging.getLogger("smoke")
log.debug("This is DEBUG (may be hidden if level is INFO)")
log.info("This is INFO (should appear)")
print("Log level used:", cfg.log_level)
