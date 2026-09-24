import unittest
import numpy as np
from .common import exact_rows,ridge_fit,shuffled_donors
class TestProtocol(unittest.TestCase):
 def test_metric_tie(self):
  r=np.asarray([[0,1,.5],[1,0,.2],[.5,.2,0.]],float);s=np.zeros_like(r);x=exact_rows(r,s);self.assertTrue(np.isfinite(x).all())
 def test_ridge(self):
  rng=np.random.default_rng(1);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2));w,b=ridge_fit(x,y,10);self.assertEqual(w.shape,(5,2));self.assertEqual(b.shape,(2,))
 def test_shuffle(self):
  u=np.asarray(list("abcdef"));d=shuffled_donors(u,list("abcd"),7);self.assertTrue(np.all(d[:4]!=np.arange(4)));self.assertTrue(set(d)<=set(range(4)))
if __name__=="__main__":unittest.main()
