"""Module for spatial checks of morphologies"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from bluepy.v2 import Circuit
import neurom as nm
from neurom.morphmath import segment_length, linear_interpolate


def _sample_morph_points(morph, sample_distance):
    """Sample points along the morphology

    Args:
        morph (neurom.FstNeuron): morphology
        sample_distance (int in um): points sampling distance

    Returns:
        Dict: sampled points per neurite. Points are of shape (N, 3) where N is the number of
        sampled points.
    """
    # map of section to its remaining offset
    section_offsets = {}
    morph_points = defaultdict(list)
    for section in nm.iter_sections(morph):
        if section.parent is None:
            parent_section_offset = 0
        else:
            parent_section_offset = section_offsets[section.parent.id]
        segment_offset = parent_section_offset
        for segment in nm.iter_segments(section):
            segment_len = segment_length(segment)
            if segment_offset + segment_len < sample_distance:
                segment_offset += segment_len
            elif segment_offset + segment_len == sample_distance:
                morph_points[section.type].append(segment[1][nm.COLS.XYZ])
                segment_offset = 0
            else:
                offsets = np.arange(sample_distance - segment_offset, segment_len, sample_distance)
                for offset in offsets:
                    morph_points[section.type].append(
                        linear_interpolate(*segment, offset / segment_len))
                segment_offset = segment_len - offsets[-1]
        section_offsets[section.id] = segment_offset
    return {neurite: np.vstack(points) for neurite, points in morph_points.items()}


def count_points_distribution(
        circuit_config, sample_count=25, sample_distance=10, mtype_random_state=0):
    """Counts distribution of morphologies points within an atlas.

    Args:
        circuit_config: any valid object for `bluepy.Circuit` constructor. For example a path to
            BlueConfig file.
        sample_count (int): number of morphologies to sample per morphology type
        sample_distance (int in um): distance between points that are sampled from a morphology
        mtype_random_state (int): seed for selecting random morphologies from the circuit

    Returns:
        Counter: a dictionary counter where keys are (mtype, soma_region, neurite, region) and
        values are counts of points within the volume designated by the key.
        (mtype, soma_region, neurite, region) designates the volume. `mtype` - morphology type,
        `soma_region` region where morphology soma is located, `neurite` - neurite of morphology
        that contains points, `region` - where points are located within the atlas. -1 means that
        they are out of the atlas bounds.
    """
    # pylint: disable=too-many-locals
    circuit = Circuit(circuit_config)
    cell_collection = circuit.cells.get()
    brain_regions = circuit.atlas.load_data('brain_regions')
    point_counter = Counter()

    for mtype in cell_collection.mtype.unique():
        mtype_exemplars = cell_collection[cell_collection.mtype == mtype]
        exemplars = mtype_exemplars.sample(
            min(sample_count, len(mtype_exemplars)), random_state=mtype_random_state)
        for example in exemplars.itertuples(index=True):
            example_morph = circuit.morph.get(example.Index, transform=True)
            neurite_points = _sample_morph_points(example_morph, sample_distance)
            for neurite, points in neurite_points.items():
                indices = brain_regions.positions_to_indices(points, False)
                within_bounds_indices = indices[np.all(indices != -1, axis=1)]
                out_bounds_count = len(points) - len(within_bounds_indices)
                regions = brain_regions.raw[tuple(within_bounds_indices.T)]
                for region, count in Counter(regions).items():
                    point_counter[(mtype, example.region, neurite, region)] += count
                point_counter[(mtype, example.region, neurite, -1)] += out_bounds_count
    return pd.DataFrame(point_counter.values(), index=pd.MultiIndex.from_tuples(
        point_counter.keys(), names=['mtype', 'soma_region', 'neurite', 'region']))
