"""A `fsspec` File System for (remote) DANDI."""
# stdlib
import re
from os import PathLike
from typing import Iterator
from urllib.parse import unquote as url_unquote

# externals
import requests
from dandi.dandiapi import (
    DandiAPIClient,
    DandiInstance,
    NotFoundError,
    RemoteAsset,
    RemoteDandiset,
)
from dandi.utils import get_instance
from fsspec.implementations.http import HTTPFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import stringify_path, tokenize


class RemoteDandiFileSystem(AbstractFileSystem):
    """
    A file system that browses through a remote dandiset.

    Examples
    --------
    Load and parse a remote file
    ```python
    from dandi.fs import RemoteDandiFileSystem
    import json
    fs = RemoteDandiFileSystem()
    with fs.open('dandi://dandi/000026/rawdata/sub-I38/ses-MRI/anat/'
                 'sub-I38_ses-MRI-echo-4_flip-4_VFA.json') as f:
        info = json.load(f)
    ```

    The 'dandi://' protocol is registered with fsspec, so the same
    result can be achived by
    ```python
    import fsspec
    import json
    with fsspec.open('dandi://dandi/000026/rawdata/sub-I38/ses-MRI/anat/'
                     'sub-I38_ses-MRI-echo-4_flip-4_VFA.json') as f:
        info = json.load(f)
    ```

    Browse a dataset
    ```python
    from dandi.fs import RemoteDandiFileSystem
    fs = RemoteDandiFileSystem('000026')
    fs.glob('**/anat/*.json')
    ```

    """

    def __init__(
        self,
        dandiset: str | RemoteDandiset | None = None,
        version: str | None = None,
        client: str | DandiInstance | DandiAPIClient | None = None,
        **http_kwargs
    ) -> None:
        """
        Initialise a remote DANDI file system.

        The root of a DANDI file system is a dandiset at a given version.
        The file system can be initialized from
            - a `RemoteDandiset` instance; or
            - the name of a dandiset [+ version]; and
                . a DandiAPIClient instance
                . a DandiInstance instance
                . the name of a known DANDI instance
                . the url of a DANDI server

        Parameters
        ----------
        dandiset : str or dandi.RemoteDandiset, optional
            An instantiated dandiset, or the identifier of a dandiset
            (e.g., `'000026'`). See `RemoteDandiset` for more info.
        version : str, optional
            The version of the dandiset to query (e.g., `'draft'`)
        client : str or dandi.DandiInstance or dandi.DandiAPIClient
            An instantiated dandi instance (or its identifier) or
            an instantiated dandi client (or its url).
            Default: `'dandi'`. See `DandiAPIClient` for more info.

        Other Parameters
        ----------------
        http_kwargs: key-value
            Any other parameters passed on to the HTTP file system
        """
        self._httpfs = HTTPFileSystem(**http_kwargs)
        super().__init__()
        if not isinstance(dandiset, RemoteDandiset):
            if isinstance(client, str):
                if not client.startswith('http'):
                    client = get_instance(client)
            if isinstance(client, DandiInstance):
                client = DandiAPIClient.for_dandi_instance(client)
            else:
                client = DandiAPIClient(client)
            if dandiset:
                dandiset = client.get_dandiset(dandiset, version)
        self._dandiset = dandiset
        self._client = None if dandiset else client

    # ------------------------------------------------------------------
    #   DANDI-specific helpers
    # ------------------------------------------------------------------

    @property
    def dandiset(self) -> RemoteDandiset:
        """Access dandiset."""
        return self._dandiset

    @dandiset.setter
    def dandiset(self, x: RemoteDandiset) -> None:
        """Assign dandiset."""
        if x:
            self._client = None
        elif self._dandiset:
            self._client = self._dandiset.client
        self._dandiset = x

    @property
    def client(self) -> DandiAPIClient:
        """Access dandi client."""
        return self.dandiset.client if self.dandiset else self._client

    @client.setter
    def client(self, x: DandiAPIClient) -> None:
        """Assign dandi client."""
        if self.dandiset:
            raise ValueError('Cannot assign a DANDI client to a FileSystem '
                             'that is already linked to a dandiset. '
                             'Unassign the dandiset first.')
        self._client = x

    @classmethod
    def for_url(cls, url: str) -> "RemoteDandiFileSystem":
        """
        Instantiate a FileSystem that interacts with the correct
        DANDI instance for a given url.
        """
        instance, dandiset, version, *_ = split_dandi_url(url)
        return cls(dandiset, version, instance)

    def get_dandiset(self, path: str) -> tuple[RemoteDandiset, str]:
        """
        If path is a relative path, return (self.dandiset, path)
        Else, the path is an absolute URL and we instantiate the correct
        remote dandiset and spit out the relative path.

        Returns: (dandiset, path) or (dandiset, asset)
        """
        dandiset = self.dandiset
        if path.startswith(('http://', 'https://', 'dandi://', 'DANDI:')):
            instance, dandiset_id, version_id, path, asset_id \
                = split_dandi_url(path)
            api_url = get_instance(instance)
            if self.client.api_url == api_url.api:
                client = self.client
            else:
                client = DandiAPIClient.for_dandi_instance(instance)
                dandiset = None
            if not asset_id:
                if not dandiset or dandiset.identifier != dandiset_id:
                    dandiset = client.get_dandiset(dandiset_id, version_id)
                if not dandiset or dandiset.version_id != version_id:
                    dandiset = client.get_dandiset(dandiset_id, version_id)
            else:
                asset = client.get_asset(asset_id)
                return dandiset, asset
        elif not self.dandiset:
            raise ValueError('File system must be linked to a dandiset to '
                             'use relative paths.')
        return dandiset, path

    def s3_url(self, path: str) -> str:
        """Get the the asset url on AWS S3."""
        dandiset, asset = self.get_dandiset(path)
        if not isinstance(asset, RemoteAsset):
            asset = dandiset.get_asset_by_path(asset)
        info = requests.request(url=asset.api_url, method='get').json()
        url = ''
        for url in info['contentUrl']:
            if url.startswith('https://dandiarchive.s3.amazonaws.com'):
                break
        if not url.startswith('https://dandiarchive.s3.amazonaws.com'):
            return None
        return url

    def _maybe_to_s3(self, url: str) -> str:
        url = stringify_path(url)
        is_s3 = url.startswith('https://dandiarchive.s3.amazonaws.com')
        # FIXME: not very generic test
        if not is_s3:
            url = self.s3_url(url)
        return url

    # ------------------------------------------------------------------
    #   FileSystem API
    # ------------------------------------------------------------------

    def ls(  # noqa: D102
        self,
        path: str | PathLike,
        detail: bool = True,
        **kwargs
    ) -> list[str] | list[dict]:
        path = stringify_path(path).strip('/')
        assets = kwargs.pop('assets', None)
        if assets is None:
            dandiset = kwargs.pop('dandiset', None)
            if not dandiset:
                dandiset, path = self.get_dandiset(path)
            assets = dandiset.get_assets_with_path_prefix(path)

        entries = []
        full_dirs = set()

        def getdate(asset: RemoteAsset, field: str) -> str:
            return getattr(getattr(asset, field, None),
                           'isoformat', lambda: None)()

        assets, assets_in = [], assets
        for asset in assets_in:
            size = getattr(asset, 'size', None)
            created = getdate(asset, 'created')
            modified = getdate(asset, 'modified')
            identifier = getattr(asset, 'identifer', None)
            asset = getattr(asset, 'path', asset)
            # 1) is the input path exactly this asset?
            asset = asset[len(path):].strip('/')
            if not asset:
                entries.append({
                    'name': path,
                    'size': size,
                    'created': created,
                    'modified': modified,
                    'identifier': identifier,
                    'type': 'file',
                })
                continue
            # 2) look at the first level under `path`
            name = asset.split('/')[0]
            fullpath = path + '/' + name
            if '/' not in asset:
                # 3) this asset is a file directly under `path`
                entries.append({
                    'name': fullpath,
                    'size': size,
                    'created': created,
                    'modified': modified,
                    'identifier': identifier,
                    'type': 'file',
                })
                continue
            else:
                # 4) this asset is a file a few levels under `path`
                # -> we do not list the path but list the directory
                if fullpath not in full_dirs:
                    entries.append({
                        'name': fullpath,
                        'size': None,
                        'type': 'directory',
                    })
                    full_dirs.add(fullpath)
            assets.append(path + '/' + asset)

        if detail:
            return entries
        else:
            return [entry['name'] for entry in entries]

    def checksum(self, path: str, **kwargs) -> str:  # noqa: D102
        # we override fsspec's default implementation when path is a
        # directory (since in this case there is no created/modified date)
        dandiset = kwargs.pop('dandiset', None)
        if not dandiset:
            dandiset, path = self.get_dandiset(path)
        assets = dandiset.get_assets_with_path_prefix(path)
        return tokenize(assets)

    def glob(  # noqa: D102
        self,
        path: str,
        order: str | None = None,
        **kwargs
    ) -> Iterator[str]:
        # we override fsspec's default implementation (which uses find)
        # to leverage the more efficient `get_assets_by_glob` from dandi
        #
        # order : [-]{created, modified, path}
        #
        # TODO: implement fsspec `maxdepth` keyword
        dandiset = kwargs.pop('dandiset', None)
        if not dandiset:
            dandiset, path = self.get_dandiset(path)
        assets = dandiset.get_assets_by_glob(path, order)
        for asset in assets:
            yield asset.path

    def exists(self, path: str, **kwargs) -> bool:  # noqa: D102
        # we override fsspec's default implementation (which uses info)
        # to avoid calls to ls (which calls get_assets_by_path on the
        # *parent* and is therefore slower)
        dandiset = kwargs.pop('dandiset', None)
        if not dandiset:
            dandiset, path = self.get_dandiset(path)
        try:
            # check if it is a file
            dandiset.get_asset_by_path(path)
            return True
        except NotFoundError:
            pass
        # check if it is a directory
        if not path.endswith('/'):
            path = path + '/'
        assets = dandiset.get_assets_with_path_prefix(path)
        try:
            next(assets)
            return True
        except StopIteration:
            return False

    def open(  # noqa: D102
        self,
        path: str,
        *args,
        **kwargs
    ) -> AbstractBufferedFile:
        path = self._maybe_to_s3(path)
        return self._httpfs.open(path, *args, **kwargs)


def split_dandi_url(url: str) -> tuple[str, str, str, str, str]:
    """
    Split a valid dandi url into its subparts.
    Returns: (instance, dandiset_id, version_id, path, asset_id)
    where instance can be an instance_id or an URL.
    """
    instance = None
    server = None
    dandiset_id = None
    version = None
    path = ''
    asset_id = None
    if url.startswith('dandi://'):
        # dandi://<instance name>/<dandiset id>[@<version>][/<path>]
        ptrn = r'dandi://(?P<i>[^/]+)/(?P<d>\d+)(@(?P<v>[^/]+))?(?P<p>.*)'
        match = re.match(ptrn, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        instance = match.group('i')
        dandiset_id = match.group('d')
        version = match.group('v')
        path = match.group('p')
    elif url.startswith(('DANDI:', 'https://identifiers.org/DANDI:')):
        # DANDI:<dandiset id>[/<version id>]
        # https://identifiers.org/DANDI:<dandiset id>[/<version id>]
        ptrn = r'(https://identifiers.org/)?DANDI:(?P<d>\d+)(/(?P<v>[^/]+))?'
        match = re.match(ptrn, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        dandiset_id = match.group('d')
        version = match.group('v')
        instance = 'DANDI'
    else:
        ptrn = r'https://(?P<s>[^/]+)(/api)?(/#)?(?P<u>.*)'
        match = re.match(ptrn, url)
        if not match:
            raise SyntaxError('Wrong dandi url')
        server = match.group('s')
        url = match.group('u')
        if url.startswith('/dandisets/'):
            # https://<server>[/api]/dandisets/<dandiset id>
            #   . [/versions[/<version>]]
            #   . /versions/<version>/assets/<asset id>[/download]
            #   . /versions/<version>/assets/?path=<path>
            ptrn = r'/dandisets/(?P<d>\d+)(/versions/(?P<v>[^/]+))?(?P<u>.*)'
            match = re.match(ptrn, url)
            if not match:
                raise SyntaxError('Wrong dandi url')
            dandiset_id = match.group('d')
            version = match.group('v')
            url = match.group('u')
            ptrn = r'/assets/((\?path=(?P<p>[.*]+))|(?P<a>[^/]+))'
            match = re.match(ptrn, url)
            if match:
                path = match.group('p')
                asset_id = match.group('a')
        elif url.startswith('/dandiset/'):
            # https://<server>[/api]/[#/]dandiset/<dandiset id>
            #   [/<version>][/files[?location=<path>]]
            ptrn = r'(/(?P<v>[^/]+))?/files(\?location=(?P<p>.*))?'
            ptrn = r'/dandiset/(?P<d>\d+)' + ptrn
            match = re.match(ptrn, url)
            dandiset_id = match.group('d')
            version = match.group('v')
            path = match.group('p')
        elif url.startswith('/assets/'):
            # https://<server>[/api]/assets/<asset id>[/download]
            ptrn = r'/assets/(?P<a>[^/]+)'
            match = re.match(ptrn, url)
            if not match:
                raise SyntaxError('Wrong dandi url')
            asset_id = match.group('a')

    path = url_unquote(path)
    path = (path or '').strip('/')

    if instance is None:
        instance = 'https://' + server

    return instance, dandiset_id, version, path, asset_id
