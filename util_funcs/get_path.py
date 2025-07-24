from pathlib import Path


def get_universal_path(relative_path):
    
    return (Path(__file__).parent / relative_path).resolve()