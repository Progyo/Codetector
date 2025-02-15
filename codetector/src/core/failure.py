from abc import ABC

class Failure(ABC):
    """
    Represents failures in the codetector framework.
    """

    def __init__(self, message:str, code:int):
        super().__init__()
        self.message = message
        """
        The message of the failure.
        """
        self.code = code
        """
        The error code of the failure.
        """

    def __eq__(self, value):
        #https://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
        if isinstance(value, self.__class__):
            return self.__dict__ == value.__dict__
        return False
    


class GenerateFailure(Failure):
    """
    Failure returned when there is an error during the generation phase.
    """
    pass



class DetectionFailure(Failure):
    """
    Failure returned when there is an error during the detection phase.
    """
    pass