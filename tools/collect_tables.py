"""Collect computed metrics into CSV without inserting manuscript numbers."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for file in sorted(a.results.rglob('metrics.json')):
        value=json.loads(file.read_text());spec=value['experiment'];metrics=value['metrics']
        rows.append(dict(table=spec['table'],experiment=spec['name'],vehicle=spec['vehicle'],infrastructure=spec['source'],representation=spec['representation'],**{k:v for k,v in metrics.items() if isinstance(v,(int,float))}))
    if not rows:raise ValueError('Point --results to completed evaluation outputs')
    columns=list(dict.fromkeys(k for row in rows for k in row));a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
