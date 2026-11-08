"""Small AlphaZero-style Connect Four: policy/value learning from PUCT self-play."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(42,64);self.policy=nn.Linear(64,7);self.value=nn.Linear(64,1)
 def forward(self,x):
  h=torch.relu(self.fc(x));return self.policy(h),torch.tanh(self.value(h)).squeeze(-1)
def winner(board):
 b=np.asarray(board).reshape(6,7)
 for r in range(6):
  for c in range(7):
   p=b[r,c]
   if not p:continue
   for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
    if 0<=r+3*dr<6 and 0<=c+3*dc<7 and all(b[r+i*dr,c+i*dc]==p for i in range(1,4)):return int(p)
 return 0
def legal(b):return [c for c in range(7) if b[c]==0]
def move(b,turn,c):
 if c not in legal(b):raise ValueError('Illegal column')
 a=b.copy()
 for r in range(5,-1,-1):
  if a[r*7+c]==0:a[r*7+c]=turn;return a
class Model:
 def __init__(self,net):self.load(net)
 def load(self,net):self.w={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
 def predict(self,b,turn):
  w=self.w;h=np.maximum(0,w['fc.weight']@(b*turn)+w['fc.bias']);log=w['policy.weight']@h+w['policy.bias'];ls=legal(b);p=np.zeros(7);exp=np.exp(log[ls]-max(log[ls]));p[ls]=exp/exp.sum();v=float(np.tanh(w['value.weight']@h+w['value.bias']).item());return p,v
class Node:
 def __init__(self,b,turn,prior=1.):self.b=b;self.turn=turn;self.prior=prior;self.n=0;self.total=0.;self.children={};self.end=winner(b);self.draw=not legal(b)
def search(b,turn,model,sims,rng=None,noise=False):
 root=Node(b,turn)
 def expand(n):
  if n.end:return -1.
  if n.draw:return 0.
  p,v=model.predict(n.b,n.turn)
  for c in legal(n.b):n.children[c]=Node(move(n.b,n.turn,c),-n.turn,float(p[c]))
  return v
 expand(root)
 if noise:
  z=rng.dirichlet(np.full(len(root.children),.3))
  for child,d in zip(root.children.values(),z):child.prior=.75*child.prior+.25*d
 for _ in range(sims):
  node=root;path=[node]
  while node.children:
   # child value is from child's perspective, negate for this player's choice
   node=max(node.children.values(),key=lambda ch:-(ch.total/ch.n if ch.n else 0)+1.5*ch.prior*math.sqrt(max(1,node.n))/(1+ch.n));path.append(node)
  value=expand(node)
  for item in reversed(path):item.n+=1;item.total+=value;value=-value
 visits=np.zeros(7)
 for c,n in root.children.items():visits[c]=n.n
 return visits/visits.sum(),root

def play(model,rng,sims):
 b=np.zeros(42,dtype=np.float32);turn=1;records=[]
 for ply in range(42):
  p,_=search(b,turn,model,sims,rng,True);records.append((b.copy(),turn,p.copy()));c=int(rng.choice(7,p=p)) if ply<10 else int(np.argmax(p));b=move(b,turn,c);w=winner(b)
  if w or not legal(b):return [(a*t,p,float(w*t)) for a,t,p in records],w,ply+1
  turn=-turn

def evaluate(model,games=40,sims=48,seed=700):
 rng=np.random.default_rng(seed);results=[]
 for game in range(games):
  agent=1 if game%2==0 else -1;b=np.zeros(42,dtype=np.float32);turn=1;cols=[]
  for ply in range(42):
   if turn==agent:p,_=search(b,turn,model,sims);c=int(np.argmax(p))
   else:c=int(rng.choice(legal(b)))
   cols.append(c);b=move(b,turn,c);w=winner(b)
   if w or not legal(b):results.append(dict(game=game,agent=agent,result=int(w*agent),moves=cols));break
   turn=-turn
 return dict(games=games,simulations=sims,seed=seed,wins=sum(x['result']==1 for x in results),draws=sum(x['result']==0 for x in results),losses=sum(x['result']==-1 for x in results),results=results)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--games',type=int,default=240);ap.add_argument('--simulations',type=int,default=24);args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(311);rng=np.random.default_rng(311);net=Net();model=Model(net);optim=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001);replay=[];history=[];start=time.time()
 initial=evaluate(model,40,48);print('initial',initial['wins'],initial['draws'],initial['losses'],flush=True)
 for game in range(args.games):
  rec,w,plies=play(model,rng,args.simulations);replay.extend(rec);replay=replay[-12000:]
  if (game+1)%20==0:
   losses=[]
   for _ in range(120):
    batch=[replay[i] for i in rng.integers(len(replay),size=min(128,len(replay)))];x=np.array([a for a,p,z in batch]);p=np.array([p for a,p,z in batch]);z=np.array([z for a,p,z in batch],dtype=np.float32);flip=rng.random(len(batch))<.5;x[flip]=x[flip].reshape(-1,6,7)[:,:,::-1].reshape(-1,42);p[flip]=p[flip,::-1];log,v=net(torch.tensor(x));pl=-(torch.tensor(p,dtype=torch.float32)*log.log_softmax(-1)).sum(-1).mean();vl=(v-torch.tensor(z)).square().mean();loss=pl+vl;optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optim.step();losses.append([pl.item(),vl.item()])
   model.load(net);history.append(dict(games=game+1,positions=len(replay),policy_loss=float(np.mean(losses,axis=0)[0]),value_loss=float(np.mean(losses,axis=0)[1])));print(game+1,history[-1],round(time.time()-start,1),flush=True)
   # Recovery checkpoint; no pickle required for browser
   Path('progress.json').write_text(json.dumps(dict(weights={k:v.tolist() for k,v in model.w.items()},history=history)))
 final=evaluate(model,60,64);fixtures=[]
 for cols in [[],[3,2,3,4,1],[0,1,0,1,0]]:
  b=np.zeros(42,dtype=np.float32);t=1
  for c in cols:b=move(b,t,c);t=-t
  p,v=model.predict(b,t);visits,_=search(b,t,model,64);fixtures.append(dict(board=b.astype(int).tolist(),turn=t,policy=p.tolist(),value=v,search=visits.tolist()))
 payload=dict(version=1,config=dict(input=42,hidden=64,seed=311,cpuct=1.5),weights={k:v.tolist() for k,v in model.w.items()},training=dict(games=args.games,simulations=args.simulations,history=history,parameters=sum(p.numel() for p in net.parameters()),root_noise=dict(alpha=.3,fraction=.25),opening_temperature_plies=10,updates_per_20_games=120),evaluation=final,initial_evaluation=initial,fixtures=fixtures)
 Path('model.json').write_text(json.dumps(payload,separators=(',',':')));print('final',final['wins'],final['draws'],final['losses'],'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
