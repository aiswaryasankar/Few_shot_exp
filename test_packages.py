import sys
print("Python version")
print (sys.version)


import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    


install('nvidia-ml-py3')

from inspect import getmembers, isfunction
import nvidia_smi


functions_list = [o for o in getmembers(nvidia_smi, isfunction) if 'Init' in o[0]]

print(functions_list) 

try:
    print('try')
    nvidia_smi.nvmlInit()
except:
    print('except')
    nvidia_smi.nvmlInit
    
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

