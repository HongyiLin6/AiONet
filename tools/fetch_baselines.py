"""Fetch pinned optional comparison sources under their upstream licenses."""
import argparse,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,default=ROOT/'workdir')
    p.add_argument('--names',nargs='+',choices=['EMIFF','v2x-vit','Select2Col'],default=['EMIFF','v2x-vit'])
    p.add_argument('--accept-select2col-terms',action='store_true',help='Confirm that you have read and can comply with the upstream NoDistrib license')
    args=p.parse_args()
    if 'Select2Col' in args.names and not args.accept_select2col_terms:p.error('Read the Select2Col upstream license before using --accept-select2col-terms')
    sources=json.loads((ROOT/'configs/baseline_sources.json').read_text());directory=args.workdir.resolve()/'src/baselines';directory.mkdir(parents=True,exist_ok=True)
    for name in args.names:
        info=sources[name];target=directory/name
        if not target.exists():subprocess.run(['git','clone',info['url'],str(target)],check=True)
        current=subprocess.run(['git','-C',str(target),'status','--porcelain'],check=True,text=True,capture_output=True).stdout
        if current:raise RuntimeError('Preserve local changes before selecting a pinned revision: '+str(target))
        subprocess.run(['git','-C',str(target),'checkout','--detach',info['commit']],check=True)
        print(name,info['commit'],info['license'])

if __name__=='__main__':main()
