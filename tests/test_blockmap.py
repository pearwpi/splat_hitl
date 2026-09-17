"""The map file the student is handed, and what a bad one looks like.

The parser here is not the one students write -- writing that is the
assignment. This one exists so `SceneBundle.check()` can hold the map up
against the scene it claims to describe, so the tests are about refusing
files that would silently mean the wrong thing.
"""
import numpy as np
import pytest

from splat_hitl.blockmap import Block, BlockMap

GOOD = """\
# a room with two boxes in it
# boundary xmin ymin zmin xmax ymax zmax
boundary -1 -2 0.3 5 2 1.8

# block xmin ymin zmin xmax ymax zmax r g b
block 1 0 0.3 2 1 1.0 255 0 0
block 3 -1 0.3 3.5 -0.5 0.8 0 255 0
"""


def _write(tmp_path, text, name="map.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_parses_a_boundary_and_its_blocks(tmp_path):
    m = BlockMap.load(_write(tmp_path, GOOD))
    assert len(m.blocks) == 2
    assert np.allclose(m.lo, [-1, -2, 0.3])
    assert np.allclose(m.hi, [5, 2, 1.8])
    assert np.allclose(m.size_m, [6, 4, 1.5])
    assert m.blocks[0].rgb == (255, 0, 0)


def test_comments_and_blank_lines_are_ignored(tmp_path):
    m = BlockMap.load(_write(tmp_path, "\n\n# nothing\nboundary 0 0 0 1 1 1 # trailing\n"))
    assert not m.blocks


def test_a_block_may_omit_its_colour(tmp_path):
    m = BlockMap.load(_write(tmp_path, "boundary 0 0 0 4 4 2\nblock 1 1 0 2 2 1\n"))
    assert m.blocks[0].rgb == (128, 128, 128)


def test_a_file_with_no_boundary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no boundary"):
        BlockMap.load(_write(tmp_path, "block 1 1 0 2 2 1 0 0 0\n"))


def test_a_second_boundary_is_refused(tmp_path):
    """Two boundaries is not a bigger room, it is a file someone concatenated."""
    with pytest.raises(ValueError, match="second boundary"):
        BlockMap.load(_write(tmp_path, "boundary 0 0 0 1 1 1\nboundary 0 0 0 2 2 2\n"))


def test_an_unknown_line_type_is_refused(tmp_path):
    """Silently skipping a line called 'sphere' would drop an obstacle."""
    with pytest.raises(ValueError, match="unknown line type"):
        BlockMap.load(_write(tmp_path, "boundary 0 0 0 4 4 2\nsphere 1 1 1 0.5\n"))


def test_a_short_block_line_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least 6"):
        BlockMap.load(_write(tmp_path, "boundary 0 0 0 4 4 2\nblock 1 1 0 2\n"))


def test_an_inverted_boundary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty or inverted"):
        BlockMap.load(_write(tmp_path, "boundary 5 5 5 1 1 1\n"))


# ------------------------------------------------------------------- queries
def test_inside_boundary_and_inside_block(tmp_path):
    m = BlockMap.load(_write(tmp_path, GOOD))
    assert m.inside_boundary([[1.5, 0.5, 0.5]])[0]
    assert not m.inside_boundary([[9.0, 0.5, 0.5]])[0]
    assert m.inside_block([[1.5, 0.5, 0.5]])[0]
    assert not m.inside_block([[4.0, 1.5, 1.0]])[0]


def test_distance_is_zero_inside_a_block_and_grows_outside(tmp_path):
    m = BlockMap.load(_write(tmp_path, GOOD))
    assert m.distance_to_obstacle([[1.5, 0.5, 0.5]])[0] == pytest.approx(0.0)
    # 0.5 m clear of the first block's +x face, level with it
    assert m.distance_to_obstacle([[2.5, 0.5, 0.5]])[0] == pytest.approx(0.5)


def test_distance_with_no_blocks_is_infinite(tmp_path):
    m = BlockMap.load(_write(tmp_path, "boundary 0 0 0 4 4 2\n"))
    assert np.isinf(m.distance_to_obstacle([[1, 1, 1]])[0])


def test_the_boundary_is_not_an_obstacle(tmp_path):
    """Leaving the room and hitting a box are different failures. Folding them
    together hides which one happened."""
    m = BlockMap.load(_write(tmp_path, "boundary 0 0 0 4 4 2\nblock 1 1 0 2 2 1\n"))
    assert m.distance_to_obstacle([[3.99, 3.99, 1.99]])[0] > 1.0


def test_sample_free_stays_out_of_the_blocks(tmp_path):
    m = BlockMap.load(_write(tmp_path, GOOD))
    g = m.sample_free(0.10)
    assert len(g) > 0
    assert not m.inside_block(g).any()
    assert m.inside_boundary(g).all()


def test_corners_are_the_eight_of_them(tmp_path):
    m = BlockMap.load(_write(tmp_path, GOOD))
    c = m.corners()
    assert c.shape == (8, 3)
    assert m.inside_boundary(c).all()


def test_a_block_knows_its_own_distance():
    b = Block(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert b.distance_to([[0.5, 0.5, 0.5]])[0] == pytest.approx(0.0)
    assert b.distance_to([[2.0, 0.5, 0.5]])[0] == pytest.approx(1.0)
    assert b.distance_to([[2.0, 2.0, 0.5]])[0] == pytest.approx(np.sqrt(2.0))
