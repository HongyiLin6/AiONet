"""Plot measured temporal metrics or the specified synthetic delay model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aionet.paper.temporal import lognormal_delays

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['temporal','delay'],required=True);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,3.5));colors=['#7299b8','#97bca9','#d9ad82','#be9cb8','#73b8b5','#cf8c77','#9b9dc7']
    if a.kind=='delay':
        values=lognormal_delays(100000,17)
        for i,name in enumerate(['Vehicle segment','Edge segment','Cloud segment','Total']):
            data=values[:,i] if i<3 else values.sum(1)
            ax.hist(data,bins=80,density=True,histtype='step',linewidth=1.6,label=name,color=colors[i])
        ax.set_xlabel('Simulated delay (ms)');ax.set_ylabel('Density')
    else:
        if a.input is None:p.error('--input is required for measured temporal plots')
        data=json.loads(a.input.read_text())['results'];keys=sorted(data,key=lambda x:float(x) if x not in ('Low','Mid','High') else ['Low','Mid','High'].index(x))
        names=list(data[keys[0]]);x=np.arange(len(keys))
        for i,name in enumerate(names):ax.plot(x,[data[k][name]['AP25'] for k in keys],marker='o',markersize=4,label=name.replace('_',' '),color=colors[i%len(colors)])
        ax.set_xticks(x);ax.set_xticklabels(keys);ax.set_xlabel('Delay regime' if keys[0] in ('Low','Mid','High') else 'Communication delay (ms)');ax.set_ylabel(r'AP$_{25}$')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8);fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=300);plt.close(fig)

if __name__=='__main__':main()
