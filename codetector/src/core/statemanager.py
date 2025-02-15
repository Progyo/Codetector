import pickle
from pathlib import Path
from random import shuffle, seed as pySeed, setstate as py_set_state, getstate as py_get_state
from numpy.random import seed as npSeed, get_state as np_get_state, set_state as np_set_state

#Inspired by: https://jamesmccaffrey.wordpress.com/2022/01/03/pytorch-training-checkpoint-exact-recovery-reproducibility/
def loadState(iterPath:str) -> int:
    """
    Load and set random number generator state from pickle file.
    """
    gen_state = {
        "iteration": 0,
        "np_random_state": None,
        "torch_random_state": None,
        "py_random_state": None,
    }

    if Path(iterPath).exists():
        with open(iterPath,'rb') as file:
            gen_state = pickle.load(file)

    iteration = gen_state['iteration']

    #Set seeds / Make reproducible
    npSeed(81053+iteration)
    if gen_state['np_random_state'] != None:
        np_set_state(gen_state['np_random_state'])
    pySeed(81053+iteration)
    if gen_state['py_random_state'] != None:
        py_set_state(gen_state['py_random_state'])
    try:
        from transformers import set_seed as t_set_seed
        t_set_seed(81053+iteration)
    except ModuleNotFoundError:
        pass
    try:
        from torch.random import manual_seed, set_rng_state
        manual_seed(81053+iteration)
        if gen_state['torch_random_state'] != None:
            set_rng_state(gen_state['torch_random_state'])
    except ModuleNotFoundError:
        pass

    return iteration

def saveState(iteration:int, iterPath:str) -> None:
    """
    Save the current iteration and random number generator state of Python, NumPy, PyTorch, Transformers.
    """
    np_random_state = np_get_state()
    py_random_state = py_get_state()

    try:
        from torch.random import get_rng_state
        torch_random_state = get_rng_state()
    except ModuleNotFoundError:
        torch_random_state = None


    gen_state = {
        "iteration": iteration,
        "np_random_state": np_random_state,
        "torch_random_state": torch_random_state,
        "py_random_state": py_random_state,
    }

    with open(iterPath,'wb') as file:
        file.write(pickle.dumps(gen_state))