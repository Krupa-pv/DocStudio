from pydantic import BaseModel, Field, ValidationError
import os

# set up the configuration from environment variables 
class Settings(BaseModel):
    #set up Azure Open AI requirements
    azure_endpoint: str = Field(..., description="AzureOpenAIEndpoint")
    azure_deployment: str = Field(..., description="AzureDeploymentName")
    azure_api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")

    #set logging level
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def load(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()
        try:
            return cls(
                azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            )
        except ValidationError as e:
            #vars are missing/invalid.
            raise RuntimeError(
                "Config error: azure open ai environment variables are missing or invalid. "
                "Set it in your environment (or a .env for local dev)."
            ) from e
        
    

