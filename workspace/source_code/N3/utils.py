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


from attrdict import AttrDict

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


def completion_callback(user_data, result, error):
    """Callback function used for async_stream_infer()."""
    user_data._completed_requests.put((result, error))


def convert_http_metadata_config(_metadata, _config):
    """Convert to the http metadata to class Dict."""
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


def requestGenerator(batched_image_data, input_name, output_name, dtype, protocol,
                     num_classes=0):
    """Generator for triton inference requests.

    Args:
        batch_image_data (np.ndarray): Numpy array of a batch of images.
        input_name (str): Name of the input array
        output_name (list(str)): Name of the model outputs
        dtype: Tensor data type for Triton
        protocol (str): The protocol used to communicated between the Triton
            server and TAO Toolkit client.
        num_classes (int): The number of classes in the network.

    Yields:
        inputs
        outputs
        made_name (str): Name of the triton model
        model_version (int): Version number
    """
    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(
            out_name, class_count=num_classes
        ) for out_name in output_name
    ]

    yield inputs, outputs
