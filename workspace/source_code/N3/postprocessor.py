# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Simple class to run post processing of Triton Inference outputs."""

import os

class Postprocessor(object):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class.

        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.        
        """
        self.batch_size = batch_size
        self.frames = frames
        self.output_path = output_path
        self.data_format = data_format
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.initialized = True

    def apply(self, output_tensors, this_id, render=True):
        """Apply the post processor to the outputs."""
        raise NotImplementedError("Base class doesn't implement any post-processing")
