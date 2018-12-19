from __future__ import print_function, division

import os
from functools import partial

from six import string_types

from ml_tools.pytils.conf import ConfigMember, OneOf
from ml_tools.pytils.conf import load_config


class DatasetsCloudConfig(ConfigMember):
    default = {
        'root': None,
        'transfer': 'sources'
    }
    validate = {
        'root': lambda root: (
            root is None
            or (
                isinstance(root, string_types)
                and root.startswith('gs://')  # only gcs supported
            )
        ),
        'transfer': OneOf('sources', 'extract')
    }


class MLDatasetsConfig(ConfigMember):
    default = {
        'root': os.path.join(os.path.expanduser('~'), 'MLTools', 'Datasets'),
        'keep_sources': False,
        'prefer_cloud': False,
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
