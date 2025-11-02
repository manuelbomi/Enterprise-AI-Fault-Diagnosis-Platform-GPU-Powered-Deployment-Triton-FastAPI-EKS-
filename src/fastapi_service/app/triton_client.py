import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

class TritonClient:
    def __init__(self, url):
        # url example: 'triton:8001'
        self.url = url
        self.client = InferenceServerClient(url=url, verbose=False)

    def infer_model(self, model_name, input_tensor):
        infer_input = InferInput(name='input__0', shape=input_tensor.shape, datatype=np_to_triton_dtype(input_tensor.dtype))
        infer_input.set_data_from_numpy(input_tensor)
        outputs = [InferRequestedOutput('embedding'), InferRequestedOutput('probs')]
        result = self.client.infer(model_name, [infer_input], outputs=outputs)
        try:
            embedding = result.as_numpy('embedding')
        except:
            embedding = input_tensor.reshape(1, -1)
        try:
            probs_arr = result.as_numpy('probs')
            probs = {'probs': probs_arr.tolist()}
        except:
            probs = {'label':'unknown', 'probs':{}}
        return embedding.astype('float32'), probs
