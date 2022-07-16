""" poisson image editing
"""
import time
import numpy as np

import fpie
from fpie.process import ALL_BACKEND, CPU_COUNT
from fpie.process import EquProcessor, GridProcessor


def poisson_image_editing(
        source, target, mask,
        backend="cuda",  # backend to use, "numpy", "numba", "taichi-cpu", "taichi-gpu", "gcc", "openmp" or "cuda"
        block_size=1024, # cuda block size (only for equ solver)
        method="equ",    # how to parallelize computation, "equ" or "grid"
        gradient="src",  # how to calculate gradient for PIE, "max", "src" or "avg"
        n_iters=5000,    # how many iteration would you perfer, the more the better
        grid_x=8, grid_y=8, # x/y axis stride for grid solver
        enable_print=True,  # print or not
    ):
    """ source.shape == (H, W, 3), uint8, ndarray
        target.shape == (H, W, 3), uint8, ndarray
        mask  .shape == (H, W),    bool,  ndarray  (ROI)
    """
    # check
    assert backend in ALL_BACKEND
    assert method in ("equ", "grid")
    assert gradient in ("max", "src", "avg")

    # print
    def smart_print(*args, **kwargs):
        if enable_print:
            print(*args, **kwargs)

    # get processor
    args_proc = (gradient, backend, CPU_COUNT, 0, block_size)
    if method == "grid":
        args_proc = (*args_proc, grid_x, grid_y)
    proc = dict(equ=EquProcessor, grid=GridProcessor)[method](*args_proc)
    
    # convert mask
    mask = mask.astype(np.uint8) * 255
    mask = mask[..., None].repeat(3, axis=2) # (H, W, 3), uint8, 0/255

    # initialize
    smart_print(
      f"Successfully initialize PIE {method} solver "
      f"with {backend} backend"
    )
    n = proc.reset(source, mask, target, (0, 0), (0, 0))
    smart_print(f"# of vars: {n}")
    proc.sync()

    # process
    t_start = time.time()
    result, err = proc.step(n_iters)
    t_end = time.time()
    smart_print(f"Abs error {err}")
    smart_print(f"Time elapsed: {t_end - t_start:.4f}s")

    return result



if __name__ == "__main__":
    """ test with the provided images
    """
    import os, sys
    import cv2
    get_path = lambda name: os.path.join(sys.argv[1], name)
    def imread_with_message(name, *args):
        path = get_path(name)
        img = cv2.imread(path, *args)
        print(f"load image from: {path}")
        return img
    def imwrite_with_message(name, img):
        path = get_path(name)
        cv2.imwrite(path, img)
        print(f"save image to: {path}")
    # load
    source = imread_with_message("in_src.png")
    target = imread_with_message("in_tar.png")
    mask   = imread_with_message("in_mask.png", 0) > 0 # to bool
    # mask target
    target *= (~mask).astype(np.uint8)[..., None] # mask out
    imwrite_with_message("out_masked_target.png", target)
    # do
    for gradient in ("src", "max", "avg"):
        out = poisson_image_editing(source, target, mask, gradient=gradient)
        imwrite_with_message(f"out_gradient={gradient}.png", out)
