from modules.layer import Layer
from modules.utils import *
# from cython_modules.im2col import im2col_forward_cython
from cython_modules.gemm_blocked import gemm_blocked
from cython_modules.gemm_omp_wrapper import gemm_omp
from numpy.lib.stride_tricks import as_strided

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct' 
        elif conv_algo == 1:
            self.mode = 'im2col'
        elif conv_algo == 2:
            self.mode = 'im2col_cython'
        elif conv_algo == 3:
            self.mode = 'im2col_omp'
        elif conv_algo == 4:
            self.mode = 'im2col_striped_omp'
        else:
            print(f"Algoritmo {conv_algo} no soportado aún")
            self.mode = 'direct' 

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        # PISTA: Usar estos if-else si implementas más algoritmos de convolución
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        elif self.mode == 'im2col_cython':
            return self._forward_im2col_cython(input)
        elif self.mode == 'im2col_omp':
            return self._forward_im2col_omp(input)
        elif self.mode == 'im2col_striped_omp':
            return self._forward_im2col_striped_omp(input)
        else:
            raise ValueError("Mode must be 'direct', 'im2col', 'im2col_cython', 'im2col_omp' or 'im2col_striped_omp'")

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c,
                                           i * self.stride:i * self.stride + k_h,
                                           j * self.stride:j * self.stride + k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]

        return output

    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # PISTA: Se te ocurren otros algoritmos de convolución?

    # Implementacion im2col 
    def _forward_im2col(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        # Hasta aqui todo igual

        columns = [] # Creamos la lista para guardar las columnas

        kernel = self.kernels.reshape(self.out_channels, -1).astype(np.float32)
    
        # Ara, para cada imagen del batch, pasamos a columnas, multiplicamos y guardamos
        for b in range(batch_size):
            columns = []
             # Pasamos los patchs a columnas
            for i in range(out_h):
                for j in range(out_w):
                    # Generado con ayuda de IA
                    patch = input[b, :, i * self.stride:i * self.stride + k_h, j * self.stride:j * self.stride + k_w]
                    columns.append(patch.reshape(-1))
                    # Fin generado con ayuda de IA

            columns = np.array(columns)   
        
            # Multiplicamos las columnas por los kernels
            # Se ha usado IA para clavar las dimensiones
            out_b = np.dot(kernel, columns.T) + self.biases.reshape(-1, 1)
            output[b] = out_b.reshape(self.out_channels, out_h, out_w)
            #Fin uso de IA

        return output

    # Im2col con Cython
    def _forward_im2col_cython(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)


        columns = [] # Creamos la lista para guardar las columnas

        kernel = self.kernels.reshape(self.out_channels, -1) # Reshapeamos los kernels a 2D
    
        # Ara, para cada imagen del batch, pasamos a columnas, multiplicamos y guardamos
        for b in range(batch_size):
            columns = []
             # Pasamos los patchs a columnas
            for i in range(out_h):
                for j in range(out_w):
                    # Generado con ayuda de IA
                    patch = input[b, :, i * self.stride:i * self.stride + k_h, j * self.stride:j * self.stride + k_w]
                    columns.append(patch.reshape(-1))
                    # Fin generado con ayuda de IA

            columns = np.array(columns, dtype=np.float32)  # float32 porque si no cython se enfada  
        
            # Hasta aqui todo igual
            # Multiplicamos las columnas por los kernels, usando la version de cython con gemm blocked
            C_out = np.zeros((self.out_channels, out_h * out_w), dtype=np.float32)
            gemm_blocked(kernel.astype(np.float32), columns.T.copy(), C_out, self.mc, self.nc, self.kc)
            # Sumamos el bias
            C_out += self.biases.reshape(-1, 1)
            output[b] = C_out.reshape(self.out_channels, out_h, out_w)

        return output
    
    def _forward_im2col_omp(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)


        columns = [] # Creamos la lista para guardar las columnas

        kernel = self.kernels.reshape(self.out_channels, -1) # Reshapeamos los kernels a 2D
    
        # Ara, para cada imagen del batch, pasamos a columnas, multiplicamos y guardamos
        for b in range(batch_size):
            columns = []
             # Pasamos los patchs a columnas
            for i in range(out_h):
                for j in range(out_w):
                    # Generado con ayuda de IA
                    patch = input[b, :, i * self.stride:i * self.stride + k_h, j * self.stride:j * self.stride + k_w]
                    columns.append(patch.reshape(-1))
                    # Fin generado con ayuda de IA

            columns = np.array(columns, dtype=np.float32)  # float32 porque si no cython se enfada  
        
            # Multiplicamos con la version de omp en C
            C_out = np.zeros((self.out_channels, out_h * out_w), dtype=np.float32)
            C_out = gemm_omp(kernel, columns.T.copy(), C_out, self.mc, self.kc, self.nc)
            # Sumamos el bias
            C_out += self.biases.reshape(-1, 1)
            output[b] = C_out.reshape(self.out_channels, out_h, out_w)

        return output
    
    def _forward_im2col_striped_omp(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)


        columns = [] # Creamos la lista para guardar las columnas

        kernel = self.kernels.reshape(self.out_channels, -1) # Reshapeamos los kernels a 2D
    
        # Construimos la matriz im2col para todo el batch de una vez usando as_strided.
        # Empleando la misma tecnica que en el maxpool2d, pero adaptada a la forma de los parches que necesitamos para im2col.
        # Generado con ayuda de IA
        sB, sC, sH, sW = input.strides

        # La idea es crear ventanas que correspondan a cada parche que necesitamos para la convolución. De forma que después
        # podemos simplemente aplanar esas ventanas y obetener las columnas que necesitamos para multiplicar por los kernels
        # directamente. 
        windows = as_strided(
            input,
            shape=(batch_size, out_h, out_w, self.in_channels, k_h, k_w),
            strides=(sB, sH * self.stride, sW * self.stride, sC, sH, sW)
        )
        # Aplanamos a (B, out_h*out_w, C_in*k_h*k_w): cada fila es un parche aplanado,
        # listo para multiplicar por los kernels como en im2col estándar.
        # np.array fuerza la copia a memoria contigua (as_strided produce vistas no contiguas).
        all_columns = np.array(windows.reshape(batch_size, out_h * out_w, self.in_channels * k_h * k_w), dtype=np.float32)
        # Fin generado con ayuda de IA
        

        # Ara, para cada imagen del batch, pasamos a columnas, multiplicamos y guardamos
        for b in range(batch_size):
            columns = all_columns[b]  # shape: (out_h*out_w, C_in*K*K)
        
            # Multiplicamos las columnas por los kernels, usando la version de cython con gemm blocked
            C_out = np.zeros((self.out_channels, out_h * out_w), dtype=np.float32)
            C_out = gemm_omp(kernel, columns.T.copy(), C_out, self.mc, self.kc, self.nc)

            # Sumamos el bias
            C_out += self.biases.reshape(-1, 1)
            output[b] = C_out.reshape(self.out_channels, out_h, out_w)

        return output