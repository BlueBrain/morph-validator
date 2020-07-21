"""Collect default feature configs."""

CONFIG_DEFAULT = {
    "neurite": {"total_length_per_neurite": ["total"]},
}

CONFIG_REPAIR = {
    "neurite": {"total_length_per_neurite": ["total"]},
}

CONFIG_SYNTHESIS = {
    "neurite": {
        "number_of_neurites": ["total"],
        "number_of_sections_per_neurite": ["mean"],
        "number_of_terminations": ["total"],
        "number_of_bifurcations": ["total"],
        "section_lengths": ["mean"],
        "section_radial_distances": ["mean"],
        "section_path_distances": ["mean"],
        "section_branch_orders": ["mean"],
        "remote_bifurcation_angles": ["mean"],
    },
    "neurite_type": ["BASAL_DENDRITE", "APICAL_DENDRITE"],
}


def get_feature_configs(config_types="default"):
    """Getter function of default features configs.

    Currently available config_types:
        - default
        - repair
        - synthesis

    Args:
        config_types (list/str): list of types of config files

    """
    if not isinstance(config_types, list):
        config_types = [config_types]
    features_config = {}
    for config_type in config_types:

        if config_type == "default":
            features_config.update(CONFIG_DEFAULT)

        if config_type == "repair":
            features_config.update(CONFIG_REPAIR)

        if config_type == "synthesis":
            features_config.update(CONFIG_SYNTHESIS)

    if not features_config:
        raise Exception("No features_config could be created with " + str(config_types))
    return features_config
