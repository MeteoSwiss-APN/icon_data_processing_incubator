import logging
import os.path
import re

logger = logging.getLogger(__name__)


def parse(def_file, component, revert=False, key_type=str):
    with open(def_file) as f:
        ll = f.readlines()
        active = False
        params = {}
        for line in ll:
            if not active:
                m = re.search(r"\'(.*)\'\s*=\s*{", line)
                if m:
                    active = True
                    paramGroup = {}
                    key_group = key_type(m[1])
                    continue
            if active:
                m = re.search(r"}", line)
                if m:
                    paramGroup = str(paramGroup)
                    if revert:
                        key_group, paramGroup = paramGroup, key_group

                    if key_group in params:
                        logger.warning(
                            f"[Component: {component}] key already inserted in DB: {key_group}"
                            + f"current value: {paramGroup}; previous value: {params[key_group]}"
                        )

                    params[key_group] = paramGroup
                    active = False
                    paramGroup = {}
            if active:
                m = re.search(r"([a-zA-Z0-9_]*)\s*=\s*(-?\d+)\s*;", line)
                if m:
                    paramGroup[m[1]] = int(m[2])

    return params


def param_db(definitions_path):
    name_file = os.path.join(definitions_path, "name.def")
    shortname_file = os.path.join(definitions_path, "shortName.def")
    paramId_file = os.path.join(definitions_path, "paramId.def")
    units_file = os.path.join(definitions_path, "units.def")

    paramId_dict = parse(paramId_file, "paramId", revert=False, key_type=int)
    shortname_dict = parse(shortname_file, "shortname", revert=True, key_type=str)
    name_dict = parse(name_file, "name", revert=True, key_type=str)
    units_dict = parse(units_file, "units", revert=True, key_type=str)

    # TODO check for duplicate vals in paramId_dict
    db = {}
    for key, val in paramId_dict.items():
        db[key] = {"params": val}
        for component in ("shortname", "name", "units"):
            if val in locals()[component + "_dict"]:
                db[key].update({component: locals()[component + "_dict"][val]})
            else:
                logger.warning(f"Key not found in {component}: {key},{val}")

    return db
