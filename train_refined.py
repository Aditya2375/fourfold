"""Fixed-budget 1,200-game PUCT self-play training. Same network, no expert games.
Resumes retain optimizer, replay, RNG and history. Only own local resume is trusted.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from train import Net,Model,play

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--chunk',type=int,default=200);args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(311);rng=np.random.default_rng(311);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);replay=[];hist=[];startgame=0;state=Path('selfplay-resume.pt')
 if state.exists():
  s=torch.load(state,weights_only=True);net.load_state_dict(s['net']);opt.load_state_dict(s['optimizer']);rng.bit_generator.state=s['rng'];replay=[(np.array(x,np.float32),np.array(p,np.float64),z) for x,p,z in s['replay']];hist=s['history'];startgame=s['games']
 model=Model(net);start=time.time();game=startgame-1
 for game in range(startgame,min(1200,startgame+args.chunk)):
  rec,w,plies=play(model,rng,64);replay.extend(rec);replay=replay[-24000:]
  if (game+1)%20==0:
   losses=[]
   for _ in range(160):
    batch=[replay[i] for i in rng.integers(len(replay),size=min(128,len(replay)))];x=np.array([a for a,p,z in batch]);p=np.array([p for a,p,z in batch]);z=np.array([z for a,p,z in batch],np.float32);flip=rng.random(len(batch))<.5;x[flip]=x[flip].reshape(-1,6,7)[:,:,::-1].reshape(-1,42);p[flip]=p[flip,::-1];log,v=net(torch.tensor(x));pl=-(torch.tensor(p,dtype=torch.float32)*log.log_softmax(-1)).sum(-1).mean();vl=(v-torch.tensor(z)).square().mean();opt.zero_grad();(pl+vl).backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append([pl.item(),vl.item()])
   model.load(net);hist.append(dict(games=game+1,positions=len(replay),policy_loss=float(np.mean(losses,0)[0]),value_loss=float(np.mean(losses,0)[1])));print(hist[-1],round(time.time()-start,1),flush=True)
   torch.save(dict(net=net.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,replay=[(x.tolist(),p.tolist(),z) for x,p,z in replay],history=hist,games=game+1),state)
 payload=dict(version=2,config=dict(input=42,hidden=64,seed=311,cpuct=1.5),weights={k:v.tolist() for k,v in model.w.items()},training=dict(games=game+1,simulations=64,history=hist,parameters=sum(p.numel() for p in net.parameters()),root_noise=dict(alpha=.3,fraction=.25),opening_temperature_plies=10,updates_per_20_games=160,learning_rate=.001,replay_cap=24000,selection='Fixed 1200-game budget, no checkpoint selection; no expert data or heuristic rescue.'),fixtures=[])
 Path('refined-model.json').write_text(json.dumps(payload,separators=(',',':')));print('saved',game+1,flush=True)
if __name__=='__main__':main()
