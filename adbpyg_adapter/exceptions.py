class ADBPyGError(Exception):
    """Base class for all exceptions in adbpyg-adapter."""


class ADBPyGValidationError(ADBPyGError, TypeError):
    """Base class for errors originating from adbpyg-adapter user input validation."""


##################
#   Metagraphs   #
##################


class ADBMetagraphError(ADBPyGValidationError):
    """Invalid ArangoDB Metagraph value"""


class PyGMetagraphError(ADBPyGValidationError):
    """Invalid PyG Metagraph value"""


##################
#   ADB -> PyG   #
##################


class ADBPyGImportError(ADBPyGError):
    """Errors on import from ArangoDB to PyG"""


class InvalidADBEdgesError(ADBPyGImportError):
    """Invalid edges on import from ArangoDB to PyG"""
