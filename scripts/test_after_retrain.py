import subprocess
import sys

def run(cmd):
    print('> ', cmd)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print('Command failed with code', res.returncode)
        sys.exit(res.returncode)

if __name__ == '__main__':
    run('python scripts/retrain_minimal.py')
    run('python run_predict_local.py')
