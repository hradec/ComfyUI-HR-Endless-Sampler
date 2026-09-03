class DirectorError(RuntimeError):
    """Base error raised by a selected local director backend."""


class DirectorDependencyError(DirectorError):
    pass


class DirectorObservationError(DirectorError):
    def __init__(self, message, *, raw_json=""):
        super().__init__(message)
        self.raw_json = raw_json


class DirectorWorkerError(DirectorObservationError):
    def __init__(self, message, *, returncode=None, raw_json=""):
        super().__init__(message, raw_json=raw_json)
        self.returncode = returncode
