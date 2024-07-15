import logging
import os
import sys
from datetime import datetime
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    container_id = os.environ.get('HOSTNAME', 'local')
    log_directory = f'logs/{timestamp}_{container_id}'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Define a custom processor to handle 'extra' fields
    def add_extra_fields(logger, method_name, event_dict):
        extra = event_dict.pop('extra', {})
        event_dict.update(extra)
        return event_dict

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_extra_fields,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler for all logs
    all_file_handler = logging.FileHandler(os.path.join(log_directory, "all.log"))
    all_file_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(all_file_handler)

    # Console handler for all logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    # Specific loggers
    logger_names = ['master', 'chat', 'memory']
    loggers = {}
    for name in logger_names:
        logger = structlog.get_logger(name)
        file_handler = logging.FileHandler(os.path.join(log_directory, f"{name}.log"))
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        loggers[name] = logger

    logging.info(f"Logging setup complete. Log directory: {log_directory}")
    return loggers

def get_logger(name):
    return structlog.get_logger(name)