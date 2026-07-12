from pathlib import Path
import time

def create_output_file_path(dir, keyname = "out"):
    path = f"{dir}/{keyname}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"
    return path

def print_to_file(message, path):
    file_path = Path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open('a', encoding='utf-8') as f:
        print(message, file=f)