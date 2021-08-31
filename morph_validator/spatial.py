"""Module for spatial checks of morphologies"""
from collections import Counter
import numpy as np
from scipy.ndimage import correlate
import pandas as pd

from bluepy import Circuit, Cell
import neurom as nm


def iter_positions(morph, sample_distance, neurite_filter=None):
    """
    Iterator for positions in space of points every <sample_distance> um.

    Assumption about segment linearity is that curvature between the start and end of segments
    is negligible.

    Args:
        morph (neurom.core.Morphology): morphology
        sample_distance (int): points sampling distance (in um)
        neurite_filter: filter neurites, see ``neurite_filter`` of ``neurom.iter_sections()``

    Yields:
        sampled points for the neurites. each point is a (3,) numpy array
    """
    section_offsets = {}

    for section in nm.iter_sections(morph, neurite_filter=neurite_filter):
        if section.parent is None:
            parent_section_offset = 0
        else:
            parent_section_offset = section_offsets[section.parent.id]
        segment_offset = parent_section_offset
        for segment in nm.iter_segments(section):
            segment_len = nm.morphmath.segment_length(segment)
            if segment_offset + segment_len < sample_distance:
                segment_offset += segment_len
            elif segment_offset + segment_len == sample_distance:
                yield segment[1][nm.COLS.XYZ]
                segment_offset = 0
            else:
                offsets = np.arange(sample_distance - segment_offset, segment_len, sample_distance)
                for offset in offsets:
                    yield nm.morphmath.linear_interpolate(*segment, offset / segment_len)
                segment_offset = segment_len - offsets[-1]
                if segment_offset == sample_distance:
                    segment_offset = 0
                    yield segment[1][nm.COLS.XYZ]
        section_offsets[section.id] = segment_offset


def _spherical_filter(radius):
    filt_size = radius * 2 + 1
    sphere = np.zeros((filt_size, filt_size, filt_size))
    center = np.array([radius, radius, radius])
    posns = np.transpose(np.nonzero(sphere == 0))
    in_sphere = posns[np.linalg.norm(posns - center, axis=-1) <= radius]
    sphere[tuple(in_sphere.transpose())] = 1
    return sphere


def relative_depth_volume(atlas, top_layer='1', bottom_layer='6',  # pylint: disable=too-many-locals
                          in_region='Isocortex', relative=True):
    """
    return volumetric data of relative cortical depth at voxel centers
    i.e. <distance from pia> / <total_cortex_thickness>
    outside of the region 'region' relative depth will be estimated,
    extrapolated from the internal relative depth
    in_region is the region within which to use the relative depth-
    values outside this region are estimated.
    """
    y = atlas.load_data("[PH]y")
    top = atlas.load_data(f"[PH]{top_layer}").raw[..., 1]
    bottom = atlas.load_data(f"[PH]{bottom_layer}").raw[..., 0]
    thickness = top - bottom
    if relative:
        reldepth = (top - y.raw) / thickness
    else:
        reldepth = y.raw
    voxel_size = y.voxel_dimensions[0]
    region = atlas.get_region_mask(in_region).raw
    to_filter = np.zeros(region.shape)
    to_filter[np.logical_and(region, reldepth < 0.5)] = -1
    to_filter[np.logical_and(region, reldepth >= 0.5)] = 1
    max_dist = 5  # voxels
    for voxels_distance in range(max_dist, 0, -1):
        filt = _spherical_filter(voxels_distance)

        num_voxels_in_range = correlate(to_filter, filt)
        # we get the estimated thickness by normalizing the filtered thickness
        # by the number of voxels that contributed
        filtered_thickness = (
            correlate(region * np.nan_to_num(thickness), filt)
            / np.abs(num_voxels_in_range))
        in_range_below = np.logical_and(num_voxels_in_range > 0.5, ~region)
        in_range_above = np.logical_and(num_voxels_in_range < -0.5, ~region)
        max_distance_from_region = voxels_distance * voxel_size
        reldepth[in_range_below] = 1 + (max_distance_from_region /
                                        filtered_thickness[in_range_below])
        reldepth[in_range_above] = -(max_distance_from_region /
                                     filtered_thickness[in_range_above])
    return y.with_data(reldepth)


def sample_morph_voxel_values(
        morphology, sample_distance, voxel_data, out_of_bounds_value, neurite_types=None):
    """
    for a specific morphology, sample the values of the neurites in voxeldata
    the value is out_of_bounds_value if the neurite is outside the voxeldata

    Arguments:
        morphology (neurom.core.Morphology): cell morphology
        sample_distance (int in um): sampling distance for neurite points
        voxel_data (voxcell.VoxelData): volumetric data to extract values from
        out_of_bounds_value: value to assign to neurites outside of voxeldata
        neurite_types (list): list of neurite types, or None (will use basal and axon)

    Returns:
        dict mapping each neurite type of the morphology to the sampled values
        {(nm.NeuriteType): np.array(...)}
    """
    if neurite_types is None:
        neurite_types = [neurite.type for neurite in morphology.neurites]

    output = {}
    for neurite_type in neurite_types:
        points = list(iter_positions(morphology, sample_distance=sample_distance,
                                     neurite_filter=lambda n, nt=neurite_type: n.type == nt))
        indices = voxel_data.positions_to_indices(points, False)
        out_of_bounds = np.any(indices == -1, axis=1)
        within_bounds = ~out_of_bounds
        values = np.zeros(len(points), dtype=voxel_data.raw.dtype)
        values[within_bounds] = voxel_data.raw[tuple(indices[within_bounds].transpose())]
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
    Get the distributions of neurites across the values of a voxel_data
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
        morph = nm.core.Morphology(circuit.morph.get(gid, transform=True))
        cell_values = sample_morph_voxel_values(
            morph, sample_distance, voxeldata, out_of_bounds_value, [nm.BASAL_DENDRITE, nm.AXON])
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
        out_of_bounds_value (int or float): value to assign to neurites outside of voxel_data
        mtype_random_state (int): seed for selecting random morphologies from the circuit

    Returns:
        pandas.DataFrame: a dataframe indexed by (mtype, soma_region, neurite, voxel_value) with a
        single column 'count' that shows number of points within the volume designated by the index.
        (mtype, soma_region, neurite, voxel_value) designates the volume. `mtype` - morphology type,
        `soma_region` region where morphology soma is located, `neurite` - neurite of morphology
        that contains points, `voxel_value` - the value of voxel_data at the neurite locations.
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
