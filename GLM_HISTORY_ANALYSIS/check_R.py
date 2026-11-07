import os
import subprocess

os.environ['R_HOME']      = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH']        = r'C:\Program Files\R\R-4.4.1\bin' + ';' + os.environ['PATH']
os.environ['R_LIBS_USER'] = r'C:\Users\alech\AppData\Local\R\win-library\4.4'

r3 = subprocess.run(['Rscript', '--version'], capture_output=True, text=True)
print("R:", r3.stdout or r3.stderr)

subprocess.run([
    'pip', 'install', '--force-reinstall', '--no-cache-dir',
    'cffi==1.17.1',
    'rpy2==3.5.11',
], check=True)