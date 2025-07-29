import os

# This script is executed by Poetry to build the CUDA extension.

# By default, we also build the SAM 2 CUDA extension.
# You may turn off CUDA build with `export SAM2_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\n"
)

def get_extensions():
    """
    Returns a list of CUDA extensions to build, or an empty list if not building CUDA.
    """
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension

        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        return [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            return []
        else:
            raise e

def build(setup_kwargs):
    """
    This function is called by Poetry to inject the build logic.
    """
    try:
        from torch.utils.cpp_extension import BuildExtension

        class BuildExtensionIgnoreErrors(BuildExtension):
            def finalize_options(self):
                try:
                    super().finalize_options()
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []

            def build_extensions(self):
                try:
                    super().build_extensions()
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []

            def get_ext_filename(self, ext_name):
                try:
                    return super().get_ext_filename(ext_name)
                except Exception as e:
                    print(CUDA_ERROR_MSG.format(e))
                    self.extensions = []
                    return "_C.so"

        # Determine which BuildExtension class to use
        builder = (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )

        setup_kwargs.update({
            "ext_modules": get_extensions(),
            "cmdclass": {"build_ext": builder},
        })
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
        else:
            raise e