import sys
from collections import Counter
import numpy as np
import pandas as pd
import bluepy.v2 as bp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from morph_validator.spatial import count_cells_points_distribution


def relative_layer_depth_volume(
        atlas, layers):
    """
    returns a VoxelData expressing relative layer depth
    This is the layer index + the proportional depth within the layer
    e.g. a cell at 1.1 is 10% of L1's thickness into L1
    a cell at 4.3 is 30% of L4's thickness from the top of L4

    Arguments:
        atlas: atlas for which to generate the volume
        layer_labels: dict of {<layer_index>: <layer_label>}

    Returns:
        VoxelData containing volumetric data of floats between 1.0
        and (len(layers) + 2) representing relative depth within layers
        at voxel centers
    """
    y = atlas.load_data('[PH]y')

    def bottom_and_top(layer):
        arr = atlas.load_data(f'[PH]{layer}').raw
        return arr[..., 0], arr[..., 1]

    output = np.zeros(y.raw.shape)
    for index, label in layers.items():
        bottom, top = bottom_and_top(label)
        proportion = (top - y.raw) / (top - bottom)
        in_layer = np.logical_and(proportion >= 0,
                                  proportion < 1)
        output[in_layer] = index + 1 + proportion[in_layer]

    return y.with_data(output)

def layer_depth_bins(num_bins_per_layer):
    """
    generate the bin edges of relative layer depth corresponding to a
    number of bins for each layer
    e.g. for [2, 3] bins for L1, L2,  edges are
    [1.0, 1.5, 2.0, 2.33, 2.66, 3.0]
    """
    return np.concatenate(
        [np.linspace(layer_num, layer_num + 1, num_bins)
         for layer_num, num_bins in num_bins_per_layer.items()])


def relative_depth_volume(atlas, top_layer='1', bottom_layer='6'):
    """
    return volumetric data of relative cortical depth at voxel centers
    i.e. <distance from pia> / <total_cortex_thickness>
    """
    y = atlas.load_data("[PH]y")
    top = atlas.load_data(f"[PH]{top_layer}").raw[..., 1]
    bottom = atlas.load_data(f"[PH]{bottom_layer}").raw[..., 0]
    thickness = top - bottom
    return y.with_data((top - y.raw) / thickness)

def sort_counter(counter):
    return pd.DataFrame(counter.values(), index=counter.keys()).sort_index()

if __name__ == "__main__":
    circuit = bp.Circuit(sys.argv[1])
    if len(sys.argv) > 2:
        layer_indices = [int(i) for i in sys.argv[2].split(",")]
        layer_labels = [i for i in sys.argv[3].split(",")]
        layer_nbins = [int(i) for i in sys.argv[4].split(",")]
    else:
        layer_indices = list(range(1, 7))
        layer_labels = [str(i) for i in layer_indices]
        layer_nbins = [8, 7, 14, 10, 20, 25]
    print("plotting a cell's overall relative layer depth neurite distribution")
    num_bins_per_layer= dict(zip(layer_indices, layer_nbins))
    layer_index_labels = dict(zip(layer_indices, layer_labels))
    neurite_counts = count_cells_points_distribution(
        circuit, [1],
        relative_layer_depth_volume(circuit.atlas, layer_index_labels),
        bin_edges=layer_depth_bins(num_bins_per_layer))
    for neurite_type, counter in neurite_counts.items():
        df = sort_counter(counter)
        plt.plot(df.values, label=str(neurite_type))
    layer_boundaries = np.cumsum(list(num_bins_per_layer.values()))
    plt.vlines(layer_boundaries, ymin=0, ymax=25)
    plt.legend()
    plt.savefig('example_single_cell.png')
    plt.show()
    plt.clf()

    mtype = 'L1_DAC'
    print("plotting the relative cortical depth distribution for mtype:", mtype)

    mtype_neurite_counts = count_cells_points_distribution(
        circuit,
        circuit.cells.ids({bp.Cell.MTYPE: mtype}, sample=50),
        relative_depth_volume(circuit.atlas),
        bin_edges=np.linspace(0, 1, 100))
    total_neurites = sum((counter for counter in mtype_neurite_counts.values()),
                         Counter())
    sorteddf = sort_counter(total_neurites)
    plt.plot(sorteddf.values)
    plt.savefig('example_mtype_sample.png')
    plt.show()
