import json,unittest
from pathlib import Path
import numpy as np,torch
from train import Net,Model,move,winner,legal,search
from puzzles import exact
class RefinedTests(unittest.TestCase):
 def test_checkpoint(self):
  m=json.loads(Path('refined-model.json').read_text());n=Net();n.load_state_dict({k:torch.tensor(v) for k,v in m['weights'].items()});net=Model(n);self.assertEqual(sum(p.numel() for p in n.parameters()),3272);self.assertEqual(m['training']['games'],1200);self.assertEqual([h['games'] for h in m['training']['history']],list(range(20,1201,20)))
  for f in m['fixtures']:
   b=np.array(f['board'],np.float32);p,v=net.predict(b,f['turn']);np.testing.assert_allclose(p,f['policy']);self.assertAlmostEqual(v,f['value']);visits,_=search(b,f['turn'],net,96);np.testing.assert_allclose(visits,f['search'])
 def test_every_proof(self):
  for p in json.loads(Path('puzzles.json').read_text()):
   b=np.array(p['board'],np.float32);self.assertFalse(winner(b));self.assertEqual(42-len(p['sequence']),p['remaining']);self.assertEqual(exact(tuple(b),p['turn']),p['outcome'])
   for o in p['options']:self.assertEqual(-exact(tuple(move(b,p['turn'],o['col'])),-p['turn']),o['value'])
 def test_match_pairing(self):
  c=json.loads(Path('comparison.json').read_text());self.assertEqual(c['games'],200);self.assertEqual(len(c['results']),200);self.assertEqual([c['wins'],c['draws'],c['losses']],[92,34,74]);g=np.random.default_rng(7621)
  for i in range(100):
   a,b=c['results'][2*i:2*i+2];self.assertEqual(a['opening'],b['opening']);self.assertEqual(a['new_side'],1);self.assertEqual(b['new_side'],-1);self.assertEqual(a['opening'],[int(g.choice(7)) for _ in range(4)])
  pairs=np.array([(c['results'][i]['result']+c['results'][i+1]['result'])/2 for i in range(0,200,2)]);self.assertAlmostEqual(pairs.mean(),c['mean_score']);g=np.random.default_rng(172);boot=np.array([g.choice(pairs,len(pairs),replace=True).mean() for _ in range(10000)]);np.testing.assert_allclose(np.quantile(boot,[.025,.975]),c['bootstrap95'])
if __name__=='__main__':unittest.main()
