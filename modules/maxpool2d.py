from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np
from numpy.lib.stride_tricks import as_strided

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    # Encapsulamos la funcion forward asi podemos elegir la optimizacion que queramos sin necesidad de otros cambios
    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        # return self.forward_original(input, training)
        return self.forward_strided(input, training)

    def forward_strided(self, input, training=True):
        self.input = input

        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1


        # Se ha empleado IA para consultar como utilizar as_strided, aunque no para crear el codigo
        # directamente. Lo comentamos igualmente por si acaso.

        # Esto nos da los pasos que tenemos que dar en memoria para movernos entre cada dimensión.
        # Es decir, para movernos de una imagen a otra, de un canal a otro, de una fila a otra, y 
        # de una columna a otra. Esto es lo que vamos a usar para crear las ventanas con as_strided.
        sB, sC, sH, sW = input.strides 

        # Creamos la ventana. 
        # La forma es (B, C, out_h, out_w, KH, KW), out_h y out_w representan el número de ventanas
        # que vamos a tener en cada dimensión, y KH y KW representan el tamaño de la ventana.
        # En strides, sB y sC se dejan igual, con sH * SH y sW * SW nos movemos entre cada ventana, y 
        # con sH y sW nos movemos dentro de la ventana.
        window = as_strided(input, shape=(B, C, out_h, out_w, KH, KW),
                            strides=(sB, sC, sH * SH, sW * SW, sH, sW))
        
        # El output es el maximo de las dos ultimas dimensiones, es decir, el maximo de cada ventana. 
        # Esto nos da un array de forma (B, C, out_h, out_w), con el maximo de cada ventana.
        output = window.max(axis=(4, 5))

        return output   

    # Funcion original
    def forward_original(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input