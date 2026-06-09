

class Task:
    def __init__(self, id: int, title: str, description: str, status: str, priority: str, due_date: str, last_update: str, **kwargs):
        """Create a new task
          id: The unique identifier of the task
          title: The title of the task
          description: The description of the task
          status: The status of the task
          priority: The priority of the task
          due_date: The due date of the task
          last_update: The last update date of the task
        """
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.due_date = due_date
        self.last_update = last_update
        #handle all the other fields with **kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert task to a dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "due_date": self.due_date,
            "last_update": self.last_update
        }
    
