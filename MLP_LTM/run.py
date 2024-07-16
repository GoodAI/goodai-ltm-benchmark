import os
import shutil
import argparse
import datetime
import uvicorn
from app.utils.logging import setup_logging, get_logger

def move_old_files_to_deprecated():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    deprecated_dir = os.path.join("_Deprecated", timestamp)
    
    os.makedirs(deprecated_dir, exist_ok=True)
    
    items_to_move = [
        "./data/memories.db",  # Update this to match your actual DB file pattern if needed
        "./logs/app.log"
    ]
    
    for item in items_to_move:
        if os.path.exists(item):
            shutil.move(item, deprecated_dir)

def main():
    parser = argparse.ArgumentParser(description="Run the application with optional fresh start.")
    parser.add_argument("--new", action="store_true", help="Move old logs and database to a deprecated folder and start fresh.")
    
    args = parser.parse_args()
    
    if args.new:
        move_old_files_to_deprecated()
    
    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Create a new empty database file if needed
    new_db_path = "./data/memories.db"
    if not os.path.exists(new_db_path):
        open(new_db_path, 'w').close()

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Starting the application.")
    
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main()
