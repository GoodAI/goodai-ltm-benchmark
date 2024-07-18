import os
import shutil
import argparse
import datetime
import uvicorn
from app.utils.logging import get_logger, setup_logger
from app.config import config
import logging

def move_old_files_to_deprecated():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    deprecated_dir = os.path.join("_Deprecated", timestamp)
    
    os.makedirs(deprecated_dir, exist_ok=True)
    
    items_to_move = [
        "./data/memories.db",
        config.LOG_FILE_MAIN,
        config.LOG_FILE_CUSTOM,
        config.LOG_FILE_CHAT
    ]
    
    for item in items_to_move:
        if os.path.exists(item):
            shutil.move(item, deprecated_dir)

def close_loggers():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

def main():
    parser = argparse.ArgumentParser(description="Run the application with optional fresh start.")
    parser.add_argument("--new", action="store_true", help="Move old logs and database to a deprecated folder and start fresh.")
    
    args = parser.parse_args()
    
    if args.new:
        # Ensure loggers are closed before moving files
        close_loggers()
        move_old_files_to_deprecated()
    
    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Create a new empty database file if needed
    new_db_path = "./data/memories.db"
    if not os.path.exists(new_db_path):
        open(new_db_path, 'w').close()

    # Setup loggers
    setup_logger('main', config.LOG_FILE_MAIN, level=config.LOG_LEVEL)
    setup_logger('custom', config.LOG_FILE_CUSTOM, level=config.LOG_LEVEL)
    setup_logger('chat', config.LOG_FILE_CHAT, level=config.LOG_LEVEL)

    logger = get_logger('custom')
    logger.info("Starting the application.")

    # Run the application with uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )

if __name__ == "__main__":
    main()
