import json
import re
from pathlib import Path
from typing import Any

def save_json_records(
    data: Any, 
    name: str, 
    subfolder: str = "", 
    base_dir: str = "data/raw",
    indent: int = 2
) -> Path:
    """
    Cleans a filename, ensures directories exist, and saves data as a JSON file.
    Designed for pandas dataframes (to_dict) or general Python lists/dicts.
    """
    
    # clean the name to be filesystem-friendly
    clean_name = re.sub(r'[ /\\]+', '_', name.lower()).strip('_')
    
    # define the full file path
    target_dir = Path(base_dir) / subfolder
    file_path = target_dir / f"{clean_name}.json"
    
    # ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # handle conversion if it's a DataFrame, otherwise save as-is
    if hasattr(data, 'to_dict'):
        records = data.to_dict(orient='records')
    else:
        records = data

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=indent, ensure_ascii=False)
        
    return file_path