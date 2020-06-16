"""Module for spatial checks of morphologies"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from bluepy.v2 import Circuit, Cell
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


def _sample_morph_voxel_values(
        morphology, sample_distance, voxeldata, out_of_bounds_value):
    """
    for a specific morphology, sample the values of the neurites in voxeldata
    the value is out_of_bounds_value if the neurite is outside the voxeldata

    Arguments:
        morphology (neurom.FstNeuron): cell morphology
        sample_distance (int in um): sampling distance for neurite points
        voxeldata (voxcell.VoxelData): volumetric data to extract values from
        out_of_bounds_value: value to assign to neurites outside of voxeldata

    Returns:
        dict mapping each neurite type of the morphology to the sampled values
        {(nm.NeuriteType): np.array(...)}
    """
    neurites_points = _sample_morph_points(morphology, sample_distance)
    output = {}
    for neurite_type, points in neurites_points.items():
        indices = voxeldata.positions_to_indices(points, False)
        out_of_bounds = np.any(indices == -1, axis=1)
        within_bounds = ~out_of_bounds
        values = np.zeros(len(points), dtype=voxeldata.raw.dtype)
        values[within_bounds] = voxeldata.raw[
            tuple(indices[within_bounds].transpose())]
        values[out_of_bounds] = out_of_bounds_value
        output[neurite_type] = values
    return output


def _count_values_in_bins(values, bin_edges=None):
    """
    count the number of elements in values in each bin described
    by bin_edges

    Arguments:
        values (np,array): array of values to count into bins
        bin_edges (list or None): the edges of the bins to count the values into. if None, simply
            count occurrences of unique values

    Returns:
        Counter of {<bin_center> : <num_occurences_within_bin>}
    """
    if bin_edges is None:
        return Counter(values)

    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) * 0.5
    bin_counts = np.histogram(values, bins=bin_edges)[0]
    return Counter(dict(zip(bin_centers, bin_counts)))


def count_cells_points_distribution(
        circuit, cell_ids, voxeldata, bin_edges=None,
        sample_distance=10, out_of_bounds_value=-1):
    """
    Get the distributions of neurites across the values of a voxeldata
    for some cells

    Arguments:
        circuit (bluepy.Circuit): circuit containing the cells
        cell_ids (list of int): cell gids e.g. (circuit.cells.ids(...))
        voxeldata (voxcell.VoxelData): containing volumetric data to use
        bin_edges: edges of the bins used to count the distribution. if None, count all occurences
            of unique values
        sample_distance (int in um): distance between points that are sampled from a morphology
        out_of_bounds_value: value to assign to neurites outside of the volume

    Returns:
        a dict of {<neurite_type> : Counter(<value>: <count>)}
    """
    counters = {}
    for gid in cell_ids:
        morph = circuit.morph.get(gid, transform=True)
        cell_values = _sample_morph_voxel_values(
            morph, sample_distance, voxeldata, out_of_bounds_value)
        for neurite_type, values in cell_values.items():
            neurite_count = _count_values_in_bins(values, bin_edges)
            if neurite_type in counters:
                counters[neurite_type] += neurite_count
            else:
                counters[neurite_type] = neurite_count
    return counters


def count_circuit_points_distribution(
        circuit_config, voxeldata=None, bin_edges=None, sample_count=25, sample_distance=10,
        out_of_bounds_value=-1, mtype_random_state=0):
    """Counts distribution of morphologies points within an atlas.

    Args:
        circuit_config: any valid object for `bluepy.Circuit` constructor. For example a path to
            BlueConfig file.
        voxeldata: a VoxelData object describing the data to count the distribution over
            if None, defaults to the 'brain_regions' data of the circuit's atlas
        bin_edges (list or None): edges of the bins used to count the distribution.
            if None, count all occurences of unique values
        sample_count (int): number of morphologies to sample per morphology type
        sample_distance (int in um): distance between points that are sampled from a morphology
        out_of_bounds_value (int or float): value to assign to neurites outside of voxeldata
        mtype_random_state (int): seed for selecting random morphologies from the circuit

    Returns:
        pandas.DataFrame: a dataframe indexed by (mtype, soma_region, neurite, voxel_value) with a
        single column 'count' that shows number of points within the volume designated by the index.
        (mtype, soma_region, neurite, voxel_value) designates the volume. `mtype` - morphology type,
        `soma_region` region where morphology soma is located, `neurite` - neurite of morphology
        that contains points, `voxel_value` - the value of voxeldata at the neurite locations.
        out_of_bounds_value means that they are out of the atlas bounds.
    """
    # pylint: disable=too-many-locals
    circuit = Circuit(circuit_config)
    if voxeldata is None:
        voxeldata = circuit.atlas.load_data('brain_regions')
        voxeldata = voxeldata.with_data(np.int32(voxeldata.raw))
    data = pd.DataFrame()
    for mtype in circuit.cells.get()[Cell.MTYPE].unique():
        mtype_cells = circuit.cells.get({Cell.MTYPE: mtype})
        exemplars = mtype_cells.sample(
            min(sample_count, len(mtype_cells)), random_state=mtype_random_state)
        for region in exemplars[Cell.REGION].unique():
            exemplars_in_region = exemplars[exemplars.region == region]
            exemplar_counts = count_cells_points_distribution(
                circuit, exemplars_in_region.index.values,
                voxeldata, sample_distance=sample_distance,
                out_of_bounds_value=out_of_bounds_value,
                bin_edges=bin_edges)
            for neurite_type, counter in exemplar_counts.items():
                data = pd.concat(
                    [data, pd.DataFrame({'mtype': mtype,
                                         'neurite': neurite_type,
                                         'soma_region': region,
                                         'voxel_value': list(counter.keys()),
                                         'count': list(counter.values())})])
    return data.set_index(['mtype', 'soma_region', 'neurite', 'voxel_value']).sort_index()
