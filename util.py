# import pandas as pd
import sys, csv, os, glob,shutil


def list_directories(path):
    """list files and directories in a given path"""
    arr = os.listdir(path)
    return arr


def clear_files(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

def del_dir(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        shutil.rmtree(f)
