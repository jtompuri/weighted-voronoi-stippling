import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _DummyAxes:
    def invert_yaxis(self):
        return None


class _DummyPyplot:
    def figure(self, *args, **kwargs):
        return None

    def scatter(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def gca(self):
        return _DummyAxes()

    def axis(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def savefig(self, *args, **kwargs):
        return None

    def close(self, *args, **kwargs):
        return None

    def show(self, *args, **kwargs):
        return None


_dummy_matplotlib = types.ModuleType("matplotlib")
_dummy_pyplot = _DummyPyplot()
_dummy_matplotlib.pyplot = _dummy_pyplot
sys.modules.setdefault("matplotlib", _dummy_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", _dummy_pyplot)

import visualize


class TestLoadTour(unittest.TestCase):
    def _write_temp(self, content: str, suffix: str = "") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix, text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except Exception:
            os.unlink(path)
            raise
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_tsplib_tour_section_basic(self):
        path = self._write_temp(
            """NAME: test\nTYPE: TOUR\nDIMENSION: 4\nTOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n""",
            suffix=".tour",
        )
        self.assertEqual(visualize.load_tour(path, expected_nodes=4), [0, 1, 2, 3])

    def test_tsplib_tour_section_multi_per_line(self):
        path = self._write_temp(
            """NAME: test\nTYPE: TOUR\nTOUR_SECTION\n1 2 3\n4 -1\n""",
            suffix=".opt.tour",
        )
        self.assertEqual(visualize.load_tour(path, expected_nodes=4), [0, 1, 2, 3])

    def test_concorde_sol_count_then_nodes_zero_based(self):
        path = self._write_temp("4\n0 1 2 3\n", suffix=".sol")
        self.assertEqual(visualize.load_tour(path, expected_nodes=4), [0, 1, 2, 3])

    def test_concorde_sol_count_then_nodes_one_based(self):
        path = self._write_temp("4\n1 2 3 4\n", suffix=".sol")
        self.assertEqual(visualize.load_tour(path, expected_nodes=4), [0, 1, 2, 3])

    def test_concorde_sol_without_explicit_count(self):
        path = self._write_temp("0 1 2 3\n", suffix=".sol")
        self.assertEqual(visualize.load_tour(path, expected_nodes=4), [0, 1, 2, 3])

    def test_concorde_sol_explicit_index_base_zero(self):
        path = self._write_temp("1 2 3\n", suffix=".sol")
        self.assertEqual(visualize.load_tour(path, index_base="0"), [1, 2, 3])

    def test_concorde_sol_explicit_index_base_one(self):
        path = self._write_temp("1 2 3\n", suffix=".sol")
        self.assertEqual(visualize.load_tour(path, index_base="1"), [0, 1, 2])

    def test_linkern_existing_format_still_works(self):
        path = self._write_temp("4\n0 1 0\n1 2 0\n2 3 0\n", suffix=".tour")
        self.assertEqual(
            visualize.load_tour(path, expected_nodes=4, tour_format="linkern"),
            [0, 1, 2, 3],
        )

    def test_linkern_closed_tour_normalized(self):
        path = self._write_temp(
            "4\n0 1 0\n1 2 0\n2 3 0\n3 0 0\n",
            suffix=".tour",
        )
        self.assertEqual(
            visualize.load_tour(path, expected_nodes=4, tour_format="linkern"),
            [0, 1, 2, 3],
        )

    def test_validation_invalid_node_id(self):
        path = self._write_temp("4\n0 1 2 4\n", suffix=".sol")
        with self.assertRaisesRegex(ValueError, "expected_nodes"):
            visualize.load_tour(path, expected_nodes=4)

    def test_validation_duplicate_node(self):
        path = self._write_temp("4\n0 1 1 3\n", suffix=".sol")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            visualize.load_tour(path, expected_nodes=4)

    def test_validation_missing_node(self):
        path = self._write_temp("4\n0 1 2 2\n", suffix=".sol")
        with self.assertRaisesRegex(ValueError, "missing"):
            visualize.load_tour(path, expected_nodes=4)

    def test_validation_length_mismatch(self):
        path = self._write_temp("0 1 2\n", suffix=".sol")
        with self.assertRaisesRegex(ValueError, "length"):
            visualize.load_tour(path, expected_nodes=4)

    def test_validation_invalid_content(self):
        path = self._write_temp("not-an-int\n", suffix=".sol")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            visualize.load_tour(path)

    def test_points_only_path_unchanged(self):
        coords = [(0.0, 0.0), (1.0, 1.0)]
        visualize.plot_points_only(coords, output_file=None, point_size=1.0)

    def test_examples_concorde_opt_tour_and_sol(self):
        tsp_path = Path("examples/filetest-concorde/random1000.tsp")
        opt_tour_path = Path("examples/filetest-concorde/filetest-concorde.opt.tour")
        sol_candidates = [
            Path("examples/filetest-concorde/filetest-concorde.sol"),
            Path("examples/filetest-concorde/random1000.sol"),
        ]
        sol_path = next((p for p in sol_candidates if p.exists()), None)
        self.assertIsNotNone(sol_path, "No Concorde .sol example file found")

        coords = visualize.load_tsp_coordinates(str(tsp_path))
        expected_nodes = len(coords)

        opt_tour = visualize.load_tour(str(opt_tour_path), expected_nodes=expected_nodes)
        sol_tour = visualize.load_tour(str(sol_path), expected_nodes=expected_nodes)

        self.assertEqual(len(opt_tour), expected_nodes)
        self.assertEqual(len(sol_tour), expected_nodes)
        self.assertEqual(set(opt_tour), set(range(expected_nodes)))
        self.assertEqual(set(sol_tour), set(range(expected_nodes)))

    def test_examples_linkern_heu_tour(self):
        tsp_path = Path("examples/filetest-linkern/random1000.tsp")
        tour_path = Path("examples/filetest-linkern/filetest-linkern.heu.tour")

        coords = visualize.load_tsp_coordinates(str(tsp_path))
        expected_nodes = len(coords)
        tour = visualize.load_tour(str(tour_path), expected_nodes=expected_nodes)

        self.assertEqual(len(tour), expected_nodes)
        self.assertEqual(set(tour), set(range(expected_nodes)))

    def test_examples_lkh_heu_tour(self):
        tsp_path = Path("examples/filetest-lkh/random1000.tsp")
        tour_path = Path("examples/filetest-lkh/filetest-lkh.heu.tour")

        coords = visualize.load_tsp_coordinates(str(tsp_path))
        tour = visualize.load_tour(str(tour_path), expected_nodes=len(coords))

        self.assertEqual(len(tour), len(coords))
        self.assertEqual(sorted(tour), list(range(len(coords))))

    def test_examples_custom_lk_heu_tour(self):
        tsp_path = Path("examples/filetest-lk/random1000.tsp")
        tour_path = Path("examples/filetest-lk/random1000.heu.tour")

        coords = visualize.load_tsp_coordinates(str(tsp_path))
        tour = visualize.load_tour(str(tour_path), expected_nodes=len(coords))

        self.assertEqual(len(tour), len(coords))
        self.assertEqual(sorted(tour), list(range(len(coords))))

    def test_sol_with_tour_section_detected_as_tsplib_in_auto_mode(self):
        path = self._write_temp(
            "NAME: x\nTYPE: TOUR\nTOUR_SECTION\n1 2 3 4\n-1\nEOF\n",
            suffix=".sol",
        )
        tour = visualize.load_tour(path, expected_nodes=4, tour_format="auto")
        self.assertEqual(tour, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
