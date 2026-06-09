import json
import os


def load_json_file(path: str):
    """Load a JSON file from a folder"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("File not found")
        return []
    except json.JSONDecodeError:
        print("Invalid JSON format")
        return []


def load_json_data(folder_path: str):
    """Load all JSON files from a folder"""
    data = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                data.append(load_json_file(file_path))
        return data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return []

def to_json(text: str):
    """Convert text to JSON by extract JSON from the text"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Invalid JSON format")
        return []