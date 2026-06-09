from datetime import datetime
from configs import settings
from utils.json_utils import load_json_data
from typing import List, Any
from langchain_core.tools import tool
import os

from models.task import Task

@tool
def get_today_task():
    """
    Get the task must done today
    """
    data_storage = load_json_data(settings.DATA_SOURCE)

    # Flatten the data_storage (which is a list of lists/dicts from JSON files)
    flat_data = []
    for file_data in data_storage:
        if isinstance(file_data, list):
            flat_data.extend(file_data)
        elif isinstance(file_data, dict):
            flat_data.append(file_data)

    # Convert to Task objects
    tasks = [Task(**task_data) for task_data in flat_data]

    today = datetime.now().strftime("%Y-%m-%d")

    def is_due_today_or_overdue(task):
        if task.due_date == today:
            return True
        # Include tasks that are overdue (due date is in the past) and not yet done
        if task.due_date != "unknown" and task.due_date < today and task.status != "done":
            return True
        return False

    return [task.to_dict() for task in tasks if is_due_today_or_overdue(task)]

@tool
def get_task_detail(task_id: str):
    """
    Get the detail of a task by its ID

    Args:
        task_id: Search for the task with this ID
    """
    data_storage = load_json_data(settings.DATA_SOURCE)

    # Flatten the data_storage (which is a list of lists/dicts from JSON files)
    flat_data = []
    for file_data in data_storage:
        if isinstance(file_data, list):
            flat_data.extend(file_data)
        elif isinstance(file_data, dict):
            flat_data.append(file_data)

    # Convert to Task objects
    tasks = [Task(**task_data) for task_data in flat_data]

    for task in tasks:
        if task.id == task_id:
            return task.to_dict()

    return None
