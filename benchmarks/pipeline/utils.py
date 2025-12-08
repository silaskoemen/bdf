import sys
import numpy as np
from loguru import logger
from sklearn.preprocessing import FunctionTransformer

def setup_logging(logging_config: dict) -> None:
    """
    Setup logging configuration using loguru.

    Parameters
    ----------
    logging_config : dict
        A dictionary containing logging configuration parameters, such as log level
    """
    # Example: You can customize the logging configuration based on the provided dictionary
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stderr, level=logging_config['level']) 

def LogTransformTransformer() -> FunctionTransformer:
    """
    Create a FunctionTransformer that applies log transformation and its inverse.
    Returns
    -------
    FunctionTransformer
        A FunctionTransformer that applies log transformation and its inverse.
    """
    return FunctionTransformer(func=lambda x: np.log(x), inverse_func=lambda x: np.exp(x))
