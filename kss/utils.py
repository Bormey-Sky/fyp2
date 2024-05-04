from typing import Dict, Any
import json

def read_config_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        return json.load(file)
        