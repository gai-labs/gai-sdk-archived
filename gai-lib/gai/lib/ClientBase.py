import os
from gai.common.utils import get_lib_config

class ClientBase:

    def __init__(self, category_name, config_path=None):
        self.category_name = category_name
        self.config = get_lib_config(config_path)

    def _get_gai_url(self):
        key = f"{self.category_name}-gai"
        config = self.config["generators"].get(key, None)
        if not config:
            raise Exception(f"Gai config does not exist. {key}")
        url = config.get("url",None)
        return url
