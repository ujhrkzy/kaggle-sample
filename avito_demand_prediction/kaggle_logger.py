# -*- coding: utf-8 -*-
import logging.config

__author__ = "ujihirokazuya"


logging_config = dict(
    version=1,
    formatters={
        'simpleFormatter': {'format': "[%(asctime)s][%(levelname)-7s][%(module)s-%(lineno)d] - %(message)s",
                            'datefmt': "%Y%m%d %H:%M:%S"}
    },
    handlers={
        'consoleHandler': {'class': 'logging.StreamHandler',
                           'formatter': 'simpleFormatter',
                           'level': logging.getLevelName(logging.DEBUG)},
        'infoHandler': {'class': 'logging.handlers.RotatingFileHandler',
                        'formatter': 'simpleFormatter',
                        'level': logging.getLevelName(logging.INFO),
                        'filename': 'kaggle_info.log',
                        'maxBytes': 1024*1024*200,
                        'backupCount': 3,
                        'encoding': 'utf-8'},
    },
    loggers={
        'loggers': {'handlers': ['consoleHandler', 'infoHandler'],
                    'level': logging.getLevelName(logging.DEBUG),
                    'propagate': True}
    }
)
logging.config.dictConfig(logging_config)
logger = logging.getLogger('loggers')
