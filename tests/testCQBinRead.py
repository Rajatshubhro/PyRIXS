import sys
sys.path.append("..")
import readCQ
import numpy as np
import pathlib

cwd = pathlib.Path.cwd() / "test.bin"
tdm = readCQ.buildCQRIXS(cwd)
print(tdm)
