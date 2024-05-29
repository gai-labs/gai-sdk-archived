import unittest
from gai.common.logging import getLogger

import os

class IT0101_logging_test(unittest.TestCase):

    def test_can_log_up_to_error(self):
        print(">>>>>>test_can_log_error:")
        os.environ['LOG_LEVEL'] = 'ERROR'
        logger = getLogger(__name__)
        logger.debug('debug message')
        logger.info('info message')
        logger.warning('warning message')
        logger.error('error message')
        logger.critical('critical message')


    def test_can_log_up_to_warning(self):
        print(">>>>>>test_can_log_warning:")
        os.environ['LOG_LEVEL'] = 'WARNING'
        logger = getLogger(__name__)
        logger.debug('debug message')
        logger.info('info message')
        logger.warning('warning message')
        logger.error('error message')
        logger.critical('critical message')

    def test_can_log_up_to_info(self):
        print(">>>>>>test_can_log_info:")
        os.environ['LOG_LEVEL'] = 'INFO'
        logger = getLogger(__name__)
        logger.info('info message')
        logger.warning('warning message')
        logger.error('error message')
        logger.critical('critical message')

    def test_can_log_up_to_debug(self):
        print(">>>>>>test_can_log_debug:")
        os.environ['LOG_LEVEL'] = 'DEBUG'
        logger = getLogger(__name__)
        logger.debug('debug message')
        logger.info('info message')
        logger.warning('warning message')
        logger.error('error message')
        logger.critical('critical message')

if __name__ == '__main__':

    unittest.main()