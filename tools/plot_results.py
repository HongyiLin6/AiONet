"""Create a publication-style metric comparison from an evaluation summary."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,default=Path('comparison.png'));p.add_argument('--policies',nargs='+');args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=json.loads(args.summary.read_text());names=args.policies or list(values)
    if any(n not in values for n in names):raise ValueError('Every policy must be present in the supplied summary')
    fig,axes=plt.subplots(1,2,figsize=(9,max(2.8,len(names)*.47)),sharey=True)
    colors=['#a5b9c6' if n in ('A0','A1','A2','A3') else '#73b8b5' if n=='R-MLP' else '#e5a16f' for n in names]
    for ax,key,title in zip(axes,['AP25','H_all'],[r'AP$_{25}$ $\uparrow$',r'Harmful-frame rate $H_{\rm all}$ $\downarrow$']):
        bars=[values[n][key] for n in names];y=np.arange(len(names));ax.barh(y,bars,color=colors,height=.55)
        ax.set_title(title,fontsize=12);ax.set_yticks(y);ax.set_yticklabels(names,fontsize=10)
        extent=max(bars+[.001]);ax.set_xlim(0,extent*1.25);ax.set_axisbelow(True);ax.grid(axis='x',color='#e2e7eb')
        for i,value in enumerate(bars):ax.text(value+extent*.015,i,f'{value:.4f}',va='center',fontsize=9)
        for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
    axes[0].invert_yaxis();fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=300,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
