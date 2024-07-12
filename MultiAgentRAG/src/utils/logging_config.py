import logging
import logging.config
import os
from datetime import datetime
from aiologger import Logger
from aiologger.handlers.files import AsyncFileHandler
from aiologger.formatters.base import Formatter
import asyncio

def setup_logging(container_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    docker_log_directory = f'/app/logs/{timestamp}_{container_id}'
    local_log_directory = f'./logs/{timestamp}_{container_id}'
    
    os.makedirs(docker_log_directory, exist_ok=True)
    os.makedirs(local_log_directory, exist_ok=True)

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'DEBUG',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'level': 'DEBUG',
                'filename': f'{docker_log_directory}/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }
    
    logging.config.dictConfig(config)
    return logging.getLogger('app')

async def setup_async_logging(docker_log_directory, local_log_directory):
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    docker_handler = AsyncFileHandler(filename=f'{docker_log_directory}/async_app.log')
    docker_handler.formatter = formatter

    local_handler = AsyncFileHandler(filename=f'{local_log_directory}/async_app.log')
    local_handler.formatter = formatter

    async_logger = Logger(name="async_logger")
    async_logger.add_handler(docker_handler)  # Remove await
    async_logger.add_handler(local_handler)   # Remove await

    return async_logger