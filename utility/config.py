import json
from pydantic import BaseModel

class Config(BaseModel):
    postgre_user: str
    postgre_password: str
    google_ai_studio_apikey: str


with open("config.json", "r", encoding="utf8") as f:
    config_data = json.load(f)

config = Config(**config_data)