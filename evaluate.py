import json
from pathlib import Path
import numpy as np,torch
from train import Net,Model,move,winner,legal,search

def run(model,kind,games=60,search_agent=True):
 rng=np.random.default_rng(990);results=[]
 for game in range(games):
  agent=1 if game%2==0 else -1;b=np.zeros(42,dtype=np.float32);turn=1;cols=[]
  for ply in range(42):
   if turn==agent:
    p,_=search(b,turn,model,64) if search_agent else (model.predict(b,turn)[0],None);c=int(np.argmax(p))
   else:
    ls=legal(b);wins=[c for c in ls if winner(move(b,turn,c))==turn];blocks=[c for c in ls if winner(move(b,-turn,c))==-turn]
    c=int(rng.choice(wins or blocks or ls)) if kind=='tactical' else int(rng.choice(ls))
   cols.append(c);b=move(b,turn,c);w=winner(b)
   if w or not legal(b):results.append(dict(game=game,agent=agent,result=int(w*agent),moves=cols));break
   turn=-turn
 return dict(seed=990,simulations=64 if search_agent else 0,games=games,wins=sum(x['result']==1 for x in results),draws=sum(x['result']==0 for x in results),losses=sum(x['result']==-1 for x in results),results=results)
if __name__=='__main__':
 torch.set_num_threads(2);torch.manual_seed(311);net=Net();untrained=Model(net);d=json.loads(Path('model.json').read_text());net.load_state_dict({k:torch.tensor(v) for k,v in d['weights'].items()});trained=Model(net);d['benchmarks']={}
 for name,model,kind,s in [('trained_tactical',trained,'tactical',True),('untrained_tactical',untrained,'tactical',True),('network_only_random',trained,'random',False)]:
  d['benchmarks'][name]=run(model,kind,search_agent=s);print(name,{k:v for k,v in d['benchmarks'][name].items() if k!='results'},flush=True)
 Path('model.json').write_text(json.dumps(d,separators=(',',':')))
