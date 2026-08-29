from parastell.energy_groups import (
    VITAMIN_J_175_EDGES_EV,
    VITAMIN_J_175_EDGES_SHA256,
    energy_edges_sha256,
)


def test_pinned_openmc_vitamin_j_175_structure_is_monotonic_and_hashable():
    assert len(VITAMIN_J_175_EDGES_EV) == 176
    assert tuple(sorted(set(VITAMIN_J_175_EDGES_EV))) == (
        VITAMIN_J_175_EDGES_EV
    )
    assert energy_edges_sha256(VITAMIN_J_175_EDGES_EV) == (
        VITAMIN_J_175_EDGES_SHA256
    )
