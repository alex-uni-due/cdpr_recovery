class Task:
    def __init__(self, func, success, failure, description) -> None:
        self.func = func
        self.success = success
        self.failure = failure
        self.description = description