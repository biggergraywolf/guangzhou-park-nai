"""Run the research workflow; check never trains models."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from package_paths import get_paths

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','preprocess','ab','comparators','m4','assemble','validate','export'])
    parser.add_argument('--output-dir',help='New output root outside this project; default: ../NAI_reproduced')
    args=parser.parse_args()
    if args.output_dir: os.environ['NAI_REPRODUCED_DIR']=args.output_dir
    paths=get_paths()
    scripts={
        'check':['check.py'],
        'preprocess':['preprocessing/merge_09_master_core_for_ml_v2.py'],
        'ab':['evaluate_models_ab.py'],
        'comparators':['evaluate_loso_comparators.py'],
        'm4':['evaluate_m4.py'],
        'assemble':['assemble_validation.py'],
        'validate':['check.py','evaluate_models_ab.py','evaluate_loso_comparators.py','evaluate_m4.py','assemble_validation.py'],
        'export':['check.py','export_urban_model.py'],
    }
    print('Project:',paths.root)
    print('New outputs:',paths.reproduced)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for filename in scripts[args.stage]:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parent/filename)],check=True,env=env)

if __name__=='__main__': main()
