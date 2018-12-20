from __future__ import print_function, division

import os
from functools import partial

from six import string_types

from ml_tools.pytils.conf import ConfigMember
from ml_tools.pytils.conf import load_config


class DatasetsCloudConfig(ConfigMember):
    default = {
        'home': None,
    }
    validate = {
        'home': lambda home: (
            home is None
            or (
                isinstance(home, string_types)
                and home.startswith('gs://')  # only gcs supported
            )
        ),
    }


class MLDatasetsConfig(ConfigMember):
    default = {
        'home': os.path.join(os.path.expanduser('~'), 'MLTools', 'Datasets'),
        'cloud': DatasetsCloudConfig()
    }


get_config = partial(
    load_config,
    config_class=MLDatasetsConfig,
    source_prio=[
        {
            'env': 'DATASET_CONFIG',
            'filepath': os.path.join(
                os.path.expanduser('~'), '.config', 'ml_tools_dataset.yaml')
        },
        {
            'env': 'ML_TOOLS_CONFIG',
            'filepath': os.path.join(
                os.path.expanduser('~'), '.config', 'ml_tools.yaml'),
            'location': 'datasets'
        }
    ],
    default=MLDatasetsConfig()
)

CONFIG = get_config()
