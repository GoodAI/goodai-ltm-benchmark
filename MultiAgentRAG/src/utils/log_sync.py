import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class LogSyncHandler(FileSystemEventHandler):
    def __init__(self, src_path, dest_path):
        self.src_path = src_path
        self.dest_path = dest_path

    def on_modified(self, event):
        if not event.is_directory:
            src_file = event.src_path
            dest_file = os.path.join(self.dest_path, os.path.basename(src_file))
            shutil.copy2(src_file, dest_file)

def start_log_sync(src_path, dest_path):
    event_handler = LogSyncHandler(src_path, dest_path)
    observer = Observer()
    observer.schedule(event_handler, src_path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_log_sync("/app/logs", "/app/local_logs")