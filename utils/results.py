import os
import re
import json

id = lambda x: x


experiments = {
    "Inductive Classifier": ("transfer", False, "inductive_adversarially_train"),
    "Inductive Detector": ("transfer_rejection_aware", True, "inductive_adversarially_train"),
    "Transductive Classifier": ("gmsa_classifier", False, None),
    "Transductive Detector": ("gmsa_tramer", True, None),
}

def get_acc(results, experiment_type=None):
    key, is_detector, subkey = experiments[experiment_type]

    summary = results[key]

    if subkey is not None:
        summary = summary[subkey]

    summary = summary["summary"]

    if is_detector:
        return summary["detector"]["tramer"]

    return summary["classifier"]


def get_rejection_rate(full_results, experiment_type=None):
    key, is_detector, subkey = experiments[experiment_type]

    if not is_detector:
        return 0

    results = full_results[key]

    if subkey is not None:
        results = results[subkey]

    return results["full"]["detector"]["tramer"]["test"]["adv_rejection_rate"]


def get_result(process=id, base_path="experimental_results", file=""):
    with open(os.path.join(base_path, file)) as f:
        value = json.load(f)
    return process(value)


def get_results(process=id, process_key=id, base_path="experimental_results", pattern="(.*).json"):
    files = os.listdir(base_path)

    matching = {}
    regex = re.compile(pattern)

    for file in files:
        match = regex.match(file)

        if match is not None:
           key = process_key(match.group(1))

           matching[key] = get_result(process=process, base_path=base_path, file=file)

    return matching

keyfinder = re.compile("{(.*?)}")

def find_results(
        pattern,
        base_path="experimental_results",
        keys={},
        process=lambda **kwargs: id,
        **kwargs
):
    remaining_keys = keyfinder.findall(pattern)

    if len(remaining_keys) == 0:
        return None

    results = {}

    replacements = {
        remaining_keys[0]: "(.*)",
    } | {
        key: ".*"
        for key in remaining_keys[1:]
    }

    first_filled = pattern.format(**replacements)

    if len(remaining_keys) == 1:
        return get_results(
            process=process(**keys),
            pattern=first_filled,
            **kwargs
        )

    first_matches = set()
    
    files = os.listdir(base_path)

    regex = re.compile(first_filled)

    for file in files:
        match = regex.match(file)

        if match is not None:
            first_matches.add(match.group(1))


    replacements = lambda v: {
        remaining_keys[0]: v,
    } | {
        key: "{" + key + "}"
        for key in remaining_keys[1:]
    }

    return {
        k: find_results(
            pattern.format(**replacements(k)),
            base_path=base_path,
            keys=keys | {remaining_keys[0]: k},
            process=process,
            **kwargs
        )
        for k in first_matches
    }
    
