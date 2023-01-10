import sys

class Logger():
    """
    A custom logger for stdout of the Python to direct output log to a file at end of runtime.
    """
    def __init__(self, filepath: str, filename: str):
        """
        Initialise both streams for stdout.

        Args:
        - filepath: str
            - Path where the file is to be made and logged
        - filename: str
            - Name of the file to be made to house the output log of the console
        """
        self.terminal = sys.stdout
        self.log = open(filepath + filename, "x")   # Open in creation mode since file does not and should not exist
   
    def write(self, message: str):
        """
        Write out the output to each stream.

        Args:
        - message: str
            - Console output to be written
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass