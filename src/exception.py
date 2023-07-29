import sys # this package is used to access many runtime variables and functions

def error_message_details(error, error_detail:sys) -> str:
    '''
    this function is used to get the error message details
    '''
    _, _, tb = sys.exc_info()
    ## file_name, line_number, func_name, text = tb.extract_tb(tb)[-1]
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    text = tb.tb_frame.f_code.co_name  
    error_message = "Error occured in python script [{0}] line number [{1}] with error [{2}]".format(file_name, line_number, text)
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        '''
        this function is used to initialize the custom exception
        '''
        self.error = error
        self.error_detail = error_detail
        self.error_message = error_message_details(error, error_detail)
        super().__init__(self.error_message)

    def __strr__(self):
        '''
        this function is used to return the error message
        '''
        return self.error_message