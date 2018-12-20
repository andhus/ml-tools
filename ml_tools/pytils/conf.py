from __future__ import print_function, division

import abc
import os
from warnings import warn

import yaml
import json


class ConfigMember(object):
    default = None
    validate = None

    def __init__(self, **kwargs):
        attrs = self.default.copy()
        for key, val in kwargs.items():
            if key not in self.default:
                raise ValueError('unexpected key: {}'.format(key))
            if isinstance(self.default[key], ConfigMember):
                attrs[key] = type(self.default[key])(**val)
            else:
                if self.validate is not None and key in self.validate:
                    if not self.validate[key](val):
                        raise ValidationError(
                            'Failed to validate: '
                            'key: {}, value: {}'.format(key, val)
                        )
                attrs[key] = val

        for attr, val in attrs.items():
            self.__setattr__(attr, val)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, filepath, location=None):
        with open(filepath) as config_f:
            config_dict = yaml.load(config_f)
        if location is not None:
            keys = location.split('.')
            for key in keys:
                config_dict = config_dict[key]
        return cls.from_dict(config_dict)

    def to_dict(self):
        config_dict = {}
        for key, val in self.default.items():
            member = getattr(self, key)
            if isinstance(member, ConfigMember):
                member = member.to_dict()
            config_dict[key] = member
        return config_dict

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            json.dumps(self.to_dict(), sort_keys=True, indent=3)
        )


class ValidationError(TypeError):
    pass


class Validator(object):

    @abc.abstractmethod
    def __call__(self, value):
        """
        # Returns:
            ok (bool)
        """


class Not(Validator):

    def __init__(self, validator):
        self.validator = validator

    def __call__(self, value):
        return not self.validator(value)


class IsInstance(Validator):

    def __init__(self, *types):
        self.types = types

    def __call__(self, value):
        return isinstance(value, self.types)


class IsNone(Validator):

    def __call__(self, value):
        return value is None


class OneOf(Validator):

    def __init__(self, *values):
        self.values = values

    def __call__(self, value):
        for compare_value in self.values:
            if value == compare_value:
                return True
            if isinstance(compare_value, Validator) and compare_value(value):
                return True

        return False

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.values)


class IterableOf(Validator):

    def __init__(self, *types):
        self.types = types

    def __call__(self, value):
        try:
            elements = list(iter(value))
        except TypeError:
            return False
        for element in elements:
            if not any([
                type(element) if isinstance(type, Validator)
                else isinstance(element, type)
                for type in self.types
            ]):
                return False
        return True


def load_config(
    config_class,
    source_prio,
    default=None
):
    """
    # Example

    CONFIG = get_config(
        MyConfig,
        [
            {'env': 'MY_CONFIG'},
            {
                'filepath': os.path.join(
                    os.path.expanduser('~'),
                    '.config',
                    'my_config.yaml'
                )
            },
            {'env': 'MY_ALT_CONFIG'},
            {
                'filepath': os.path.join(
                    os.path.expanduser('~'),
                    '.config', 'my_alt_config.yaml'
                ),
                'location': 'internal.path.to.my_config'
            },
        ],
        MyConfig()
    )

    """
    config = None
    for source in source_prio:
        if 'env' in source.keys():
            config_path = os.getenv(source['env'], source.get('filepath', None))
            if config_path and os.path.isfile(config_path):
                config = config_class.from_yaml(
                    config_path,
                    location=source.get('location', None)
                )
        elif 'filepath' in source.keys():
            if os.path.isfile(source['filepath']):
                config = config_class.from_yaml(
                    source['filepath'],
                    location=source.get('location', None)
                )
    if config is None:
        if default is not None:
            config = default
            warn('No config was found, using default: {}'.format(config))
        else:
            raise ValueError(
                'could not find config, verify sources or provide a default value'
            )

    return config
