import os
from doc_studio.core.config import Settings


cfg = Settings.load()
print("Azure Config OK:")
print("Endpoint:", cfg.azure_endpoint)
print("Deployment:", cfg.azure_deployment)
print("API Version:", cfg.api_version)
