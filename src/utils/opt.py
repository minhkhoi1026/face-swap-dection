"""
Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py
"""
from typing import Optional
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import json
from src.utils.loading import load_yaml


class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        print(f"Saving config to {path}...")
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        print(f"Loading config from {path}...")
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts(ArgumentParser):
    def __init__(self, cfg: Optional[str] = None):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-c", "--config", default=cfg, help="configuration file to use"
        )
        self.add_argument(
            "-o", "--opt", nargs="+", help="override configuration options"
        )

    def parse_args(self, argv=None):
        args = super(Opts, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)

        config = Config(args.config)
        config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        print("Overriding configurating")
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    print(f"'{key}' not found in config")
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0]
                )
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            print(f"'{key}' not found in config")
                    else:
                        cur = cur[sub_key]
        return global_config

def parse_config(yaml_file, mode: str="raw"):
    config = Config(yaml_file)
    
    if (mode == "raw"):
        return config
    
    if (mode == "wandb"):
        # remove _wandb config
        config.pop("_wandb")
        config.pop("wandb_version")
        # for each key, assign it equal to its "value" key
        for key in config.keys():
            config[key] = config[key]["value"]
        return config
    
def compare(d1, d2, path=""):
    if (type(d1) != type(d2)):
        print(f"Type mismatch between two instance!, d1: {type(d1)}, d2: {type(d2)}")
        return

    if isinstance(d1, dict):
        for k in d1:
            if k not in d2:
                print(f"Key '{path}.{k}' exists in the first dictionary but not the second.")
            else:
                compare(d1[k], d2[k], f"{path}.{k}")
    elif isinstance(d1, list):
        if len(d1) != len(d2):
            print(f"Array length mismatch for key '{path}': {len(d1)} != {len(d2)}")
        else:
            for i in range(len(d1)):
                compare(d1[i], d2[i], f'{path}.[{i}]')
    else:
        if (d1!= d2):
            print(f"Value mismatch for key '{path}': {d1} != {d2}")


if __name__ == "__main__":
    import sys
    print(parse_config(sys.argv[1], mode="wandb"))
    compare(parse_config(sys.argv[1], mode="wandb"), parse_config("configs/single_head.yml"))
