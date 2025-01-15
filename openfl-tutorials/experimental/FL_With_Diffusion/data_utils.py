import numpy as np
from functools import partial


def features_labels_to_dict(features, labels):
    return {'features': features, 'labels': labels}


def dict_to_features_labels(_dict):
    return _dict['features'], _dict['labels']


def test_probably_equal(dict1, dict2, tag=""):
    features1, labels1 = dict_to_features_labels(_dict=dict1)
    features2, labels2 = dict_to_features_labels(_dict=dict2)
    f1 = np.sum(features1.astype(np.float64)).item()
    f2 = np.sum(features2.astype(np.float64)).item()
    assert np.abs(f1 - f2) < 0.0001, f"For test: {tag}, sums of two features sets from dicts are not equal ({f1} versus {f2})."
    l1 = np.sum(labels1.astype(np.float64)).item()
    l2 = np.sum(labels2.astype(np.float64)).item()
    assert np.abs(l1 - l2) < 0.0001, f"For test {tag} sums of two labels sets from dicts are not equal ({l1} versus {l2})."

def counts_across_dicts(*dicts):
    """
    dicts (data dicts)
    Return: Counts of the data size of each dict in the list of inputs
    """
    return list(map(lambda x: len(dict_to_features_labels(x)[1]), dicts))


def combine_dicts(*dicts, shuffle, shuffle_seed=None):
    """
    appends the arrays withing the 'features' and 'labels' keys of all dicts provided
    """
    features, labels = map(partial(np.concatenate, axis=0), zip(*map(dict_to_features_labels, dicts)))
    _dict = features_labels_to_dict(features=features, labels=labels)

    if shuffle:
        _dict = shuffle_data_within_dict(_dict=_dict, shuffle_seed=shuffle_seed)
    return _dict


def shuffle_data_within_dict(_dict, shuffle_seed=None):
    if shuffle_seed:
        rng = np.random.default_rng(shuffle_seed) 
    else:
        rng = np.random.default_rng()

    features, labels = dict_to_features_labels(_dict=_dict)

    indices = np.arange(len(features))
    rng.shuffle(indices)
    shuffled_dict = features_labels_to_dict(features=features[indices], labels=labels[indices])
    return shuffled_dict


def split_data_by_class(_dict):
    """
    _dict (dict): data dict having keys 'features' and 'labels' each with numpy array  values

    Returns: Dict of class to inner dict, inner dict taking 'features' and 'labels' to arrays
    """
    dict_by_class = {}
    features, labels = dict_to_features_labels(_dict=_dict)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_mask = (labels==label)
        dict_by_class[label] = features_labels_to_dict(features=features[label_mask], labels=labels[label_mask])
    return dict_by_class


def split_off_classes(target_classes, dict_by_class):
    num_classes = len(dict_by_class.keys())
    left_over = dict_by_class
    split_off = {}

    for target_class in target_classes:
        split_off[target_class] = left_over.pop(target_class)

    assert len(split_off) + len(left_over) == num_classes, f"Got {len(split_off)} split off classes with {len(left_over)} left over when {num_classes} classes were provided to the split function."

    return split_off, left_over


def combine_data_over_classes(dict_by_class, shuffle=False, shuffle_seed=None):
    # Currently not used
    """
    dict_by_class (dict): class to inner dict, inner dict taking 'features' and 'labels' to features array and labels array respectively

    Returns: features, labels tuple of arrays 
    """
    combined_dicts = combine_dicts(dict_by_class.values())
    if shuffle:
        combined_dicts = shuffle_data_within_dict(_dict=combined_dicts, shuffle_seed=shuffle_seed)
    return combined_dicts
        

def stratified_split(dict_by_class, n_parts, shuffle=True, shuffle_seed=None):
    """
    Returns: dict with keys of split index and values dicts with keys 'features' and 'labels' with array values.
    """
    if shuffle:
        dict_by_class = {label: shuffle_data_within_dict(_dict=dict_by_class[label], shuffle_seed=shuffle_seed) for label in dict_by_class}
    split_dict = {}
    counts_by_class_by_split = {}
    for label, dict_for_label in dict_by_class.items():
        features, labels = dict_to_features_labels(_dict=dict_for_label)
        counts_by_class_by_split[label] = {}
        for idx in range(n_parts):
            part_features = features[idx::n_parts]
            part_labels = labels[idx::n_parts]
            part_dict = features_labels_to_dict(features=part_features, labels=part_labels)
            counts_by_class_by_split[label][idx] = len(part_features)
            
            # double check
            assert set(part_labels) == set([label]), f"labels part as a set is: {set(part_labels)} when {set([label])} was expected."
            
            if idx not in split_dict:
                split_dict[idx] = part_dict
            else:
                split_dict[idx] = combine_dicts(split_dict[idx], part_dict, shuffle=shuffle, shuffle_seed=shuffle_seed)
    
    # double check we didn't arive a different set of total data
    combined_split_dict = combine_dicts(*split_dict.values(), shuffle=False)
    combined_dict_by_class = combine_dicts(*dict_by_class.values(), shuffle=False)
    test_probably_equal(dict1=combined_split_dict, dict2=combined_dict_by_class)

    # double check that the sizes of the splits are within reasonable range of eachother (use labels)
    split_counts = counts_across_dicts(*split_dict.values())
    assert max(split_counts) - min(split_counts) < len(dict_by_class) + 1, f"Split counts {split_counts} appear to be suspicious."

    return split_dict, counts_by_class_by_split
