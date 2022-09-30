# Copyright (c) 2022 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, 
# under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials include references to hardware and software 
# developed by other entities; all applicable licensing and copyrights apply.

import gdown
import os
import shutil

url = "https://drive.google.com/uc?id=1E8KaSkexo5U4OhiDIrfUipBbHcvwWCvJ&export=download"
output = "dataset_E2ECV.zip"
gdown.download(url, output, quiet=False, proxy=None)

shutil.unpack_archive(output)

if not os.path.exists("../data"):
    shutil.move("data", "../data")
else:
    shutil.rmtree("data")

if not os.path.exists("../source_code/N4/apples.h264"):
    shutil.move("apples.h264", "../source_code/N4/apples.h264")
else:
    os.remove("apples.h264")

if not os.path.exists("../source_code/N5/oranges.mp4"):
    shutil.move("oranges.mp4", "../source_code/N5/oranges.mp4")
else:
    os.remove("oranges.mp4")

if not os.path.exists("../source_code/N5/oranges"):
    shutil.move("oranges", "../source_code/N5/oranges")
else:
    shutil.rmtree("oranges")

os.remove(output)

