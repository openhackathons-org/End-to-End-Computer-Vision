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

"""Triton inference client for TAO Toolkit model."""


import tritonclient.grpc.model_config_pb2 as mc
import numpy as np


class TritonModel(object):
    """Simple class to run model inference using Triton client."""

    def __init__(self, max_batch_size, input_names, output_names,
                 channels, height, width, data_format, triton_dtype):
        """Set up a detectnet_v2 triton model instance.
        
        Args:
            max_batch_size(int): The maximum batch size of the TensorRT engine.
            input_names (str): List of the input node names
            output_names (str): List of the output node names
            channels (int): Number of chanels in the input dimensions
            height (int): Height of the input
            width (int): Width of the input
            data_format (str): The input dimension order. This can be "channels_first"
                or "channels_last". "channels_first" is in the CHW order,
                and "channels_last" is in HWC order.
            triton_dtype (proto): Triton input data type.
            channel_mode (str): String order of the C dimension of the input.
                "RGB" or "BGR"
                
        Returns:
            An instance of the DetectnetModel.
        """
        self.max_batch_size = max_batch_size
        self.input_names = input_names
        self.output_names = output_names
        self.c = channels
        assert channels in [1, 3], (
            "TAO Toolkit models only support 1 or 3 channel inputs."
        )
        self.h = height
        self.w = width
        self.data_format = data_format
        self.triton_dtype = triton_dtype
        self.scale = 1
        if channels == 3:
            self.mean = [0., 0., 0.]
        else:
            self.mean = [0]
        self.mean = np.asarray(self.mean).astype(np.float32)
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.mean = self.mean[:, np.newaxis, np.newaxis]

    @staticmethod
    def parse_model(model_metadata, model_config):
        """Simple class to parse model metadata and model config."""
        raise NotImplementedError("Base class doesn't implement this method.")

    @classmethod
    def from_metadata(cls, model_metadata, model_config):
        """Parse a model from the metadata config."""
        parsed_outputs = cls.parse_model(model_metadata, model_config)
        max_batch_size, input_names, output_names, channels, height, width, \
            data_format, triton_dtype = parsed_outputs
        return cls(
            max_batch_size, input_names, output_names,
            channels, height, width, data_format,
            triton_dtype
        )

    def get_config(self):
        """Get dictionary config."""
        config_dict = {
            "data_format": self.data_format,
            "max_batch_size": self.max_batch_size,
            "channels": self.c,
            "width": self.w,
            "height": self.h,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "triton_dtype": self.triton_dtype
        }
        return config_dict

    def preprocess(self, image):
        """Function to preprocess image

        Performs mean subtraction and then normalization.

        Args:
            image (np.ndarray): Numpy ndarray of an input batch.

        Returns:
            image (np.ndarray): Preprocessed input image.
        """
        image = (image - self.mean) * self.scale
        return image
