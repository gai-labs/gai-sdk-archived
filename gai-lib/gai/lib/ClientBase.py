import os
from gai.common.utils import get_lib_config

class ClientBase:

    def __init__(self, config_path=None):
        if config_path:
            self.config = get_lib_config(config_path)
        else:
            self.config = get_lib_config()
        self.base_url = self.config["gai_url"]

    def _gen_url(self, generator):
        url = os.path.join(self.base_url,
                           self.config["generators"][generator]["url"].lstrip('/'))
        return url
