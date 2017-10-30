from kernels import *

class KernelFactory():

    @staticmethod
    def create_kernel(kernel_name, *args, **kwargs):
        if kernel_name == 'pol':
            return PolynomialKernel(*args, **kwargs)
        if kernel_name == 'rbf':
            return RBFKernel(*args, **kwargs)
        if kernel_name == 'laplace':
            return LaplacianKernel(*args, **kwargs)
        if kernel_name == 'tanh':
            return HyperbolicTangentKernel(*args, **kwargs)
        return BaseKernel(*args, **kwargs)
