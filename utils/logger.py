import sys

"""
A custom logger for stdout of the Python to direct output log to a file at end of runtime.
"""
class Logger():
    def __init__(self, filepath: str, filename: str, verbose: bool) -> None:
        """
        Initialise both streams for stdout.

        Args:
        - filepath: str
            - Path where the file is to be made and logged
        - filename: str
            - Name of the file to be made to house the output log of the console
        - verbose: bool
            - Controls whether terminal is written to or not
        """
        self.terminal = sys.stdout
        self.log = open(filepath + filename, "x", encoding="utf-8")   # Open in creation mode since file does not and should not exist
        self.verbose = verbose
   

    def write(self, message: str) -> None:
        """
        Write out the output to each stream.

        Args:
        - message: str
            - Console output to be written
        """
        if self.verbose:
            self.terminal.write(message)
        self.log.write(message)


    def flush(self):
        pass