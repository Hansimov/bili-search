from copy import deepcopy


def get_es_source_val(d: dict, key: str):
    keys = key.split(".")
    dd = deepcopy(d)
    for key in keys:
        if isinstance(dd, dict) and key in dd:
            dd = dd[key]
        else:
            joined_key = ".".join(keys)
            if joined_key in dd:
                return dd[joined_key]
            else:
                return None

    return dd


if __name__ == "__main__":
    d = {
        "owner": {"name": {"space": "value1"}},
        "stat": {"rights": {"view": "value2"}},
        "pubdate.time": "value3",
    }

    k1 = "owner.name.space"
    k2 = "stat.rights.view"
    k3 = "pubdate.time"
    for k in [k1, k2, k3]:
        print(f"{k}: {get_es_source_val(d,k)}")

    # python -m elastics.structure
