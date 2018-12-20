from __future__ import print_function, division

import os
from functools import partial

from ml_tools.pytils.conf import ConfigMember, load_config
from ml_tools.dataset.config import MLDatasetsConfig


class OutputConfig(ConfigMember):
    default = {
        'root': os.path.join(os.path.expanduser('~'), 'MLTools', 'Output')
    }


class MLToolsConfig(ConfigMember):
    default = {
        'datasets': MLDatasetsConfig(),
        'output': OutputConfig()
    }


get_config = partial(
    load_config,
    config_class=MLToolsConfig,
    source_prio=(
        {'env': 'ML_TOOLS_CONFIG'},
        {'filepath': os.path.join(
            os.path.expanduser('~'), '.config', 'ml_tools.yaml')}
    ),
    default=MLToolsConfig()
)

CONFIG = get_config()
