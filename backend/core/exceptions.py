class StockPredictorException(Exception):
    """Base exception for the application"""
    pass

class DataNotFoundException(StockPredictorException):
    """Raised when stock data file is not found"""
    def __init__(self, message="Stock data not found"):
        self.message = message
        super().__init__(self.message)

class ModelNotTrainedException(StockPredictorException):
    """Raised when inference is attempted without a trained model"""
    def __init__(self, message="Model not found. Please train first."):
        self.message = message
        super().__init__(self.message)

class PreprocessingException(StockPredictorException):
    """Raised when data preprocessing fails"""
    pass
