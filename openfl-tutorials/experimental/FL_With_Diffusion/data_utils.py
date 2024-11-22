import numpy as np
from functools import partial


def features_labels_to_dict(features, labels):
    return {'features': features, 'labels': labels}


def dict_to_features_labels(_dict):
    return _dict['features'], _dict['labels']


def test_probably_equal(dict1, dict2):
    features1, labels1 = dict_to_features_labels(_dict=dict1)
    features2, labels2 = dict_to_features_labels(_dict=dict2)
    assert np.sum(features1).item() == np.sum(features2).item(), f"Sums of two features sets from dicts are not equal."
    assert np.sum(labels1).item() == np.sum(labels2).item(), f"Sums of two labels sets from dicts are not equal."

def counts_across_dicts(*dicts):
    """
    dicts (data dicts)
    Return: Counts of the data size of each dict in the list of inputs
    """
    return map(lambda x: len(dict_to_features_labels[1]), *dicts)


def combine_dicts(*dicts):
    """
    appends the arrays withing the 'features' and 'labels' keys of all dicts provided
    """
    features, labels = map(partial(np.concatenate, axis=0), zip(*map(dict_to_features_labels, dicts)))
    return features_labels_to_dict(features=features, labels=labels)


def shuffle_data_within_dict(_dict, shuffle_seed=None):
    if shuffle_seed:
        rng = np.random.default_rng(shuffle_seed) 
    else:
        rng = np.random.default_rng()

    features, labels = dict_to_features_labels(_dict=_dict)
    
    # keep features synced with their labels during shuffling
    _data = list(zip(features, labels))
    rng.shuffle(_data)
    shuffled_features, shuffled_labels = map(np.array, zip(*_data))
    return features_labels_to_dict(features=shuffled_features, labels=shuffled_labels)


def split_data_by_class(_dict):
    """
    _dict (dict): data dict having keys 'features' and 'labels' each with numpy array  values

    Returns: Dict of class to inner dict, inner dict taking 'features' and 'labels' to arrays
    """
    features, labels = dict_to_features_labels(_dict=_dict)
    dict_by_class = {}
    for feature, label in zip(features, labels):
        dict_to_append = features_labels_to_dict(features=np.expand_dims(feature, axis=0), labels=np.expand_dims(label, axis=0))
        if label not in dict_by_class:
            dict_by_class[label] = dict_to_append
        else:
            dict_by_class[label] = combine_dicts(dict_by_class[label], dict_to_append)
    return dict_by_class


def combine_data_over_classes(dict_by_class, shuffle=False, shuffle_seed=None):
    """
    dict_by_class (dict): class to inner dict, inner dict taking 'features' and 'labels' to features array and labels array respectively

    Returns: features, labels tuple of arrays 
    """
    combined_dicts = combine_dicts(dict_by_class.values())
    if shuffle:
        shuffle_data_within_dict(_dict=combined_dicts, shuffle_seed=shuffle_seed)
    return combined_dicts
        

def stratified_split(dict_by_class, n_parts, shuffle=True, shuffle_seed=None):
    """
    Returns: dict with keys of split index and values dicts with keys 'features' and 'labels' with array values.
    """
    if shuffle:
        dict_by_class = {label: shuffle_data_within_dict(_dict=dict_by_class[label], shuffle_seed=shuffle_seed) for label in dict_by_class}

    split_dict = {}
    # split_dict = {idx: {'features': [], 'labels': []} for idx in range(n_parts)}
    for label, dict_for_label in dict_by_class.items():
        features, labels = dict_to_features_labels(_dict=dict_for_label)
        for idx in range(n_parts):
            part_features = features[idx::n_parts]
            part_labels = labels[idx::n_parts]
            part_dict = features_labels_to_dict(features=part_features, labels=part_labels)
            
            # double check
            assert set(part_labels) == set([label]), f"labels part as a set is: {set(part_labels)} when {set([label])} was expected."
            
            if idx not in split_dict:
                split_dict[idx] = part_dict
            else:
                split_dict[idx] = combine_dicts(split_dict[idx], part_dict)
    
    # double check we didn't arive a different set of total data
    combined_split_dict = combine_dicts(*split_dict.values())
    combined_dict_by_class = combine_dicts(*dict_by_class.values())
    test_probably_equal(dict1=combined_split_dict, dict2=combined_dict_by_class)

    # double check that the sizes of the splits are within reasonable range of eachother (use labels)
    split_counts = counts_across_dicts(*split_dict.values())
    assert max(split_counts) - min(split_counts) < n_parts + 1, f"Split counts {split_counts} appear to be suspicious."

    return split_dict
