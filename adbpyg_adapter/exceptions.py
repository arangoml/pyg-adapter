class ADBPyGError(Exception):
    """Base class for all exceptions in adbpyg-adapter."""


class ADBPyGValidationError(ADBPyGError, TypeError):
    """Base class for errors originating from adbpyg-adapter user input validation.

    :param msg: Error message.
    :type msg: str
    """

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.message = msg


##################
#   Metagraphs   #
##################


class ADBMetagraphError(ADBPyGValidationError):
    """Invalid ArangoDB Metagraph value"""


class PyGMetagraphError(ADBPyGValidationError):
    """Invalid PyG Metagraph value"""
