"""Adversarial timing tests; synthetic inputs, no experimental scores."""
import unittest
from aionet.feature_bank import Observation,FeatureBank,serial_source_ready

def obs(name,o,a,scenario='s',family='F'):
    return Observation(name,scenario,'roadside',family,o,a,a,'cache/'+name,'pose/'+name)

class BankTests(unittest.TestCase):
    def test_out_of_order_arrival_keeps_newest_observation(self):
        bank=FeatureBank();bank.reset('s')
        bank.advance(400000,[obs('new',200000,300000)])
        first=bank.snapshot('roadside','F',350000)
        bank.advance(500000,[obs('late_old',100000,450000)])
        self.assertEqual(bank.snapshot('roadside','F',450000).latest.packet_id,'new')
        self.assertEqual(len(first.real),1)
    def test_excludes_future_relative_to_target_even_if_arrived(self):
        bank=FeatureBank();bank.reset('s');bank.advance(300000,[obs('future',250000,260000)])
        self.assertIsNone(bank.snapshot('roadside','F',200000).latest)
    def test_forecast_uses_only_three_real_arrived_frames(self):
        bank=FeatureBank();bank.reset('s')
        bank.advance(500000,[obs(str(t),t,t+100000) for t in (100000,200000,300000)])
        snap=bank.snapshot('roadside','F',400000)
        self.assertEqual(len(snap.prediction_history()),3)
        bank.advance(600000,[obs('4',400000,600000)])
        self.assertEqual(snap.latest.observation_us,300000)
        self.assertNotEqual(snap.fingerprint,bank.snapshot('roadside','F',500000).fingerprint)
    def test_rejects_future_packet_cross_scene_and_duplicate_mutation(self):
        bank=FeatureBank();bank.reset('s')
        with self.assertRaises(ValueError):bank.advance(1,[obs('x',0,2)])
        with self.assertRaises(ValueError):bank.advance(3,[obs('x',0,2,'other')])
        bank.advance(3,[obs('x',0,2)])
        with self.assertRaises(ValueError):bank.advance(4,[obs('x',1,2)])
        with self.assertRaises(ValueError):bank.advance(2)
        bank.reset('next');bank.advance(0)
        self.assertIsNone(bank.snapshot('roadside','F',0).latest)
    def test_gapped_history_is_ineligible(self):
        bank=FeatureBank();bank.reset('s')
        bank.advance(600000,[obs(str(t),t,t) for t in (0,100000,500000)])
        self.assertEqual(bank.snapshot('roadside','F',600000).prediction_history(),())
    def test_source_queue_does_not_deliver_uncomputed_features(self):
        self.assertEqual(serial_source_ready([0,100,200],[150,150,50]),[150,300,350])
        self.assertEqual(serial_source_ready([0,100,200],[50,50,50]),[50,150,250])
    def test_rollout_cap_matches_predictor_for_nonuniform_clock(self):
        bank=FeatureBank();bank.reset('s')
        bank.advance(2200000,[obs(str(t),t,t) for t in (0,95000,190000)])
        self.assertEqual(bank.snapshot('roadside','F',2140000).prediction_history(),())
        self.assertEqual(len(bank.snapshot('roadside','F',2090000).prediction_history()),3)

if __name__=='__main__':unittest.main()
