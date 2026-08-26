from types import SimpleNamespace

import numpy as np
import pytest

from scripts.regenerate_selected_magnet_source_cad import (
    select_original_filament,
)


class MagnetSet:
    def __init__(self):
        self.filaments = []
        self.magnet_coils = []

    def _instantiate_filaments(self):
        self.filaments = [
            SimpleNamespace(coords=np.asarray([[index, 0.0, 0.0]]))
            for index in range(3)
        ]

    def populate_magnet_coils(self):
        self._instantiate_filaments()
        self.magnet_coils = [
            SimpleNamespace(coords=self.filaments[index].coords)
            for index in (2, 0)
        ]


def test_selection_tracks_original_identity_through_filter_and_sort():
    magnet_set = MagnetSet()
    original_count, filtered_index = select_original_filament(magnet_set, 0)

    assert original_count == 3
    assert filtered_index == 1
    assert len(magnet_set.magnet_coils) == 1
    assert magnet_set.magnet_coils[0].coords[0, 0] == 0


def test_selection_rejects_source_filament_removed_by_sector_filter():
    with pytest.raises(ValueError, match="absent or ambiguous"):
        select_original_filament(MagnetSet(), 1)
