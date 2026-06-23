import sys


def error_message_detail(error, error_detail: sys):
    """Extract detailed error information from the traceback."""
    exc_info = error_detail.exc_info()
    if exc_info[2] is not None:
        exc_tb = exc_info[2]
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = (
            f"Error occurred in [{file_name}] line [{exc_tb.tb_lineno}]: {str(error)}"
        )
    else:
        error_message = f"Error: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = None):
        super().__init__(error_message)
        if error_detail:
            self.error_message = error_message_detail(error_message, error_detail=error_detail)
        else:
            self.error_message = str(error_message)

    def __str__(self):
        return self.error_message
