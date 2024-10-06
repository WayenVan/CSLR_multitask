import sys
from logging import Logger
import gc
import torch
from pathlib import Path


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def info(l: Logger, m):
    if l is not None:
        l.info(m)


def warn(l: Logger, m):
    if l is not None:
        l.warning(m)


def add_attributes(obj, locals: dict):
    for key, value in locals.items():
        if key != "self" and key != "__class__":
            setattr(obj, key, value)


def is_namedtuple_instance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(field, str) for field in fields)


def is_debugging():
    # Check if the script is executed with the -d or --debug option
    if "-d" in sys.argv or "--debug" in sys.argv:
        return True

    # Check if the script is executed with the -O or -OO option
    if sys.flags.optimize:
        return False

    # Check if the script is executed with the -X or --pdb option
    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        return True

    # Check if the Python debugger module is imported
    if "pdb" in sys.modules:
        return True

    return False


def clean_folder(folder_path):
    folder = Path(folder_path)

    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove file or symbolic link
            elif item.is_dir():
                for sub_item in item.rglob("*"):  # Recursively remove contents
                    if sub_item.is_file() or sub_item.is_symlink():
                        sub_item.unlink()
                    elif sub_item.is_dir():
                        sub_item.rmdir()
                item.rmdir()  # Finally remove the empty directory
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")
