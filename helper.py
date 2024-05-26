import torch
import gc
import time
from functools import partial
import pandas as pd

time_hook_dict = {}
time_pre_hook_dict = {}
iterations = {}

times_dict = {}
list_passes_pre_hooks = []
list_pre_hooks = []
list_hooks = []


def clear_cuda(**kwargs):
    print(f"initial cuda usage {torch.cuda.memory_allocated()}")
    for k, v in kwargs.items():
        del(v)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"final cuda usage {torch.cuda.memory_allocated()}")
    return None

def passes_in_hook(module, input, name):
    iterations[name] += 1
    return None


def time_pre_hook(module, input, times_dict, name):
    time_pre_hook_dict[name] = time.perf_counter()
    return None 

def time_hook(module, input, output, times_dict, name):
    time_hook_dict[name] = time.perf_counter()
    if name not in times_dict:
        times_dict[name] = time_hook_dict[name] - time_pre_hook_dict[name]
    else:
        times_dict[name] += time_hook_dict[name] - time_pre_hook_dict[name]
    return None

def timeit(module, name=None):
    if name is None:
        name = module.__class__.__name__
    passes_hook = partial(passes_in_hook, name=name)
    pre_hook = partial(time_pre_hook, times_dict=times_dict, name=name)
    hook = partial(time_hook, times_dict=times_dict, name=name)
    list_pre_hooks.append(module.register_forward_pre_hook(pre_hook))
    list_hooks.append(module.register_forward_hook(hook))
    list_passes_pre_hooks.append(module.register_forward_pre_hook(passes_hook))

    time_hook_dict[name] = 0
    time_pre_hook_dict[name] = 0
    iterations[name] = 0

    return None

def hook_children(model):
    for name, module in model.named_children():
        timeit(module, name)
    return None

def run_with_time(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
    print_times()
    for key, value in times_dict.items():
        times_dict[key] = 0
        iterations[key] = 0
    return output

def remove_all_hooks():
    for pre_hook in list_pre_hooks:
        pre_hook.remove()
    for hook in list_hooks:
        hook.remove()
    return None

def print_times():
    # Create a DataFrame from the times dictionary
    df = pd.DataFrame({
        'Module': list(times_dict.keys()),
        'Time (s)': list(times_dict.values()),
        'Iterations': [iterations[name] for name in times_dict.keys()],
        'Avg Time per Iteration (s)': [times_dict[name] / iterations[name] for name in times_dict.keys()]
    })
    # Sort the DataFrame by time
    df = df.sort_values(by='Time (s)', ascending=False)
    # Print the DataFrame
    print(df)

