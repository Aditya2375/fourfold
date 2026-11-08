"""Fixed-seed paired opening matches: same PUCT budget, swapped colors.
No solved-game strength or Elo claim. A shared search algorithm is not an independent opponent.
"""
import json
from pathlib import Path
import numpy as np,torch
from train import Net,Model,move,winner,legal,search

def load(name):
 d=json.loads(Path(name).read_text());n=Net();n.load_state_dict({k:torch.tensor(v) for k,v in d['weights'].items()});return Model(n)
def main():
 torch.set_num_threads(2);old=load('model.json');new=load('refined-model.json');g=np.random.default_rng(7621);results=[];start=0
 if Path('benchmark-progress.json').exists():
  saved=json.loads(Path('benchmark-progress.json').read_text());results=saved['results'];start=len(results)//2;g.bit_generator.state=saved['rng']
 for seed in range(start,100):
  b=np.zeros(42,np.float32);t=1;opening=[]
  for _ in range(4):
   c=int(g.choice(legal(b)));opening.append(c);b=move(b,t,c);t=-t
  for side in [1,-1]:
   board=b.copy();turn=t;moves=opening.copy()
   while not winner(board) and legal(board):
    m=new if turn==side else old;p,_=search(board,turn,m,96);c=int(np.argmax(p));moves.append(c);board=move(board,turn,c);turn=-turn
   results.append(dict(opening_id=seed,opening=opening,new_side=side,result=winner(board)*side,moves=moves))
  Path('benchmark-progress.json').write_text(json.dumps(dict(results=results,rng=g.bit_generator.state)))
  if seed%10==9:print('completed',len(results),flush=True)
 # Pair bootstrap by opening, preserving color-swap correlation.
 pairs=np.array([(results[i]['result']+results[i+1]['result'])/2 for i in range(0,len(results),2)]);rng=np.random.default_rng(172);boot=np.array([rng.choice(pairs,len(pairs),replace=True).mean() for _ in range(10000)])
 payload=dict(seed=7621,opening_plies=4,openings=100,games=200,simulations=96,wins=sum(r['result']==1 for r in results),draws=sum(r['result']==0 for r in results),losses=sum(r['result']==-1 for r in results),mean_score=float(pairs.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_seed=172,bootstrap_resamples=10000,results=results)
 Path('comparison.json').write_text(json.dumps(payload,separators=(',',':')));print({k:v for k,v in payload.items() if k!='results'})
 # Inference and exact search reference fixtures for the new frozen checkpoint.
 d=json.loads(Path('refined-model.json').read_text());d['fixtures']=[]
 for cols in [[],[3,2,3,4,1],[0,1,0,1,0],[6,6,5,5,4,4]]:
  b=np.zeros(42,np.float32);t=1
  for c in cols:b=move(b,t,c);t=-t
  p,v=new.predict(b,t);visits,root=search(b,t,new,96);d['fixtures'].append(dict(board=b.astype(int).tolist(),turn=t,policy=p.tolist(),value=v,search=visits.tolist()))
 Path('refined-model.json').write_text(json.dumps(d,separators=(',',':')))
if __name__=='__main__':main()
