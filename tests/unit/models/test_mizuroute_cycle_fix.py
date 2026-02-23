
import logging
import unittest

import numpy as np

from symfluence.models.mizuroute.topology_generator import MizuRouteTopologyGenerator


class TestMizuRouteCycleFix(unittest.TestCase):
    def setUp(self):
        # Mock config and logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.DEBUG)

        # Create topology generator with a mock preprocessor
        class MockPreprocessor:
            logger = self.logger
        self.generator = MizuRouteTopologyGenerator(MockPreprocessor())

    def test_simple_cycle(self):
        """Test breaking a simple 3-node cycle: 1 -> 2 -> 3 -> 1"""
        # Node IDs: 1, 2, 3
        seg_ids = np.array([1, 2, 3])
        # Flow: 1->2, 2->3, 3->1
        down_seg_ids = np.array([2, 3, 1])
        # Elevations: 1=100m, 2=90m, 3=95m
        # Node 2 is the lowest, so it should become the outlet (downSegId=0)
        elevations = np.array([100.0, 90.0, 95.0])

        fixed_down = self.generator._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        # Check that node 2 (index 1) now points to 0
        self.assertEqual(fixed_down[1], 0)
        # Others should remain unchanged
        self.assertEqual(fixed_down[0], 2)
        self.assertEqual(fixed_down[2], 1)

    def test_cycle_with_tail(self):
        """Test cycle with a tail: 4 -> 1 -> 2 -> 1"""
        # Nodes: 1, 2, 4
        seg_ids = np.array([1, 2, 4])
        # Flow: 1->2, 2->1, 4->1
        down_seg_ids = np.array([2, 1, 1])
        # Elevations: 1=100, 2=50, 4=150
        # Node 2 is lowest in cycle, should break
        elevations = np.array([100.0, 50.0, 150.0])

        fixed_down = self.generator._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        self.assertEqual(fixed_down[1], 0) # Node 2 breaks
        self.assertEqual(fixed_down[0], 2) # Node 1 still points to 2
        self.assertEqual(fixed_down[2], 1) # Node 4 still points to 1

    def test_complex_multi_cycle(self):
        """Test multiple disconnected cycles"""
        # Cycle A: 1 <-> 2
        # Cycle B: 3 <-> 4
        seg_ids = np.array([1, 2, 3, 4])
        down_seg_ids = np.array([2, 1, 4, 3])
        # Elevations
        # 1=100, 2=90 (break 2)
        # 3=50, 4=60 (break 3)
        elevations = np.array([100.0, 90.0, 50.0, 60.0])

        fixed_down = self.generator._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        self.assertEqual(fixed_down[1], 0) # Node 2 breaks
        self.assertEqual(fixed_down[2], 0) # Node 3 breaks

        self.assertEqual(fixed_down[0], 2)
        self.assertEqual(fixed_down[3], 3) # Node 4 points to 3 (now outlet)

    def test_no_cycle(self):
        """Test linear path with no cycle"""
        seg_ids = np.array([1, 2, 3])
        down_seg_ids = np.array([2, 3, 0])
        elevations = np.array([100.0, 90.0, 80.0])

        fixed_down = self.generator._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        np.testing.assert_array_equal(fixed_down, down_seg_ids)

if __name__ == '__main__':
    unittest.main()
