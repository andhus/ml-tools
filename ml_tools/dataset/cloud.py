from __future__ import print_function, division


try:
    from tensorflow.python.lib.io import file_io as tf_file_io
except ImportError:
    tf_file_io = None

try:
    from google.cloud import storage as gcs
except ImportError:
    storage = None


class DatasetNotAvailable(IOError):
    pass


gcs_prefix = 'gs://'


class CloudIOError(IOError):
    pass


def get_gcs_bucket_and_object_name(uri):
    assert uri.startswith(gcs_prefix)
    splits = uri.split('/')  # [gs:, '', 'bucket', 'path' 'to', 'target']
    bucket_name = splits[2]
    object_name = '/'.join(splits[3:])
    return bucket_name, object_name


def _gcs_copy(source_filepath, target_filepath):
    """Copies a file to/from/within Google Cloud Storage (GCS).
    # Arguments
        source_filepath: String, path to the file on filesystem or object on GCS to
            copy from.
        target_filepath: String, path to the file on filesystem or object on GCS to
            copy to.
        overwrite: Whether we should overwrite an existing file/object at the target
            location, or instead ask the user with a manual prompt.
    """
    if tf_file_io is None:
        raise ImportError('Google Cloud Storage file transfer requires TensorFlow.')
    with tf_file_io.FileIO(source_filepath, mode='rb') as source_f:
        with tf_file_io.FileIO(target_filepath, mode='wb') as target_f:
            target_f.write(source_f.read())


def save_to_cloud(source_path, target_uri):
    if target_uri.startswith(gcs_prefix):
        # prefer TensorFlow option
        if tf_file_io is not None:
            _gcs_copy(source_path, target_uri)
            return

        if gcs is None:
            raise ImportError('you must have google.cloud.storage installed')
        bucket_name, object_name = get_gcs_bucket_and_object_name(target_uri)
        client = gcs.Client()
        bucket = client.get_bucket(bucket_name)
        print(
            'uploading {} to GCS bucket: {}, object: {}'.format(
                source_path, bucket_name, object_name
            )
        )
        blob = bucket.blob(object_name)
        blob.upload_from_filename(source_path)
    else:
        raise ValueError('cloud uri not supported')


def load_from_cloud(source_uri, target_path):
    if source_uri.startswith(gcs_prefix):
        # prefer TensorFlow option
        if tf_file_io is not None:
            _gcs_copy(source_uri, target_path)
            return

        if gcs is None:
            raise ImportError('you must have google.cloud.storage installed')
        bucket_name, object_name = get_gcs_bucket_and_object_name(source_uri)
        client = gcs.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(object_name)
        if blob.exists():
            print(
                'loading GCS bucket: {}, object: {} to {}'.format(
                    bucket_name, object_name, target_path
                )
            )
            blob.download_to_filename(target_path)
        else:
            raise CloudIOError('object does not exists')
    else:
        raise ValueError('cloud target not supported')
