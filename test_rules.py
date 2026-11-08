import unittest,json
import numpy as np
from train import winner,move,legal,Model,Net,search
class Checks(unittest.TestCase):
 def test_gravity(self):
  b=np.zeros(42);b=move(b,1,3);self.assertEqual(b[38],1);b=move(b,-1,3);self.assertEqual(b[31],-1)
 def test_full_column(self):
  b=np.zeros(42)
  for i in range(6):b=move(b,1 if i%2==0 else -1,0)
  self.assertNotIn(0,legal(b))
  with self.assertRaises(ValueError):move(b,1,0)
 def test_lines(self):
  for start,step in [(35,1),(0,7),(0,8),(6,6)]:
   b=np.zeros(42)
   for i in range(4):b[start+i*step]=1
   self.assertEqual(winner(b),1)
 def test_no_wrap(self):
  b=np.zeros(42);b[[5,6,7,8]]=1;self.assertEqual(winner(b),0)
 def test_immediate_win(self):
  b=np.zeros(42)
  for c,t in [(0,1),(1,-1),(0,1),(1,-1),(0,1)]:b=move(b,t,c)
  p,root=search(b,1,Model(Net()),192);self.assertEqual(int(np.argmax(p)),0)
 def test_export(self):
  d=json.loads(__import__('pathlib').Path('model.json').read_text());self.assertEqual(d['training']['games'],240);self.assertEqual(len(d['evaluation']['results']),60)
if __name__=='__main__':unittest.main()
