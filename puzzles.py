"""Generate legal late-game positions and exact exhaustive outcomes. No model training uses these."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from train import move,winner,legal
@lru_cache(None)
def exact(board,turn):
 b=np.array(board,np.float32);w=winner(b)
 if w:return int(w*turn)
 if not legal(b):return 0
 best=-1
 for c in legal(b):
  value=-exact(tuple(move(b,turn,c)), -turn);best=max(best,value)
  if best==1:break
 return best

def main():
 rng=np.random.default_rng(9871);puzzles=[];attempts=0
 while len(puzzles)<12 and attempts<10000:
  attempts+=1;b=np.zeros(42,np.float32);t=1;cols=[]
  for ply in range(32+attempts%4):
   choices=[c for c in legal(b) if not winner(move(b,t,c))]
   if not choices:break
   c=int(rng.choice(choices));cols.append(c);b=move(b,t,c);t=-t
  if len(cols)<32:continue
  options=[dict(col=c,value=-exact(tuple(move(b,t,c)),-t)) for c in legal(b)];outcome=max(o['value'] for o in options)
  # Keep a mix, including non-winning endings and positions with different outcomes by move.
  if len(puzzles)<6 and len(set(o['value'] for o in options))<2:continue
  puzzles.append(dict(id=len(puzzles)+1,name='Endgame '+str(len(puzzles)+1),sequence=''.join(str(c+1) for c in cols),board=b.astype(int).tolist(),turn=t,remaining=42-len(cols),outcome=outcome,options=options))
 Path('puzzles.json').write_text(json.dumps(puzzles,separators=(',',':')));print('puzzles',len(puzzles),'attempts',attempts,'states',exact.cache_info())
if __name__=='__main__':main()
