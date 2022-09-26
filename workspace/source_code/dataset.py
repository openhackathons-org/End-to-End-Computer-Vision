# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved. 

import gdown
import os
import shutil

url = "https://drive.google.com/uc?id=1E8KaSkexo5U4OhiDIrfUipBbHcvwWCvJ&export=download"
output = "dataset_E2ECV.zip"
gdown.download(url, output, quiet=False, proxy=None)

shutil.unpack_archive(output)
os.remove(output)

shutil.move("data", "../data")
shutil.move("apples.h264", "N4/apples.h264")
shutil.move("oranges.mp4", "N5/oranges.mp4")
shutil.move("oranges", "N5/oranges")

