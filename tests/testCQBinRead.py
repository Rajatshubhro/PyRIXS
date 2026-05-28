import sys
sys.path.append("..")
import readCQTDM
import numpy as np
import pathlib

cwd = pathlib.Path.cwd() / "test.bin"
tdm = readCQTDM.buildTDM(cwd)
print(tdm)
