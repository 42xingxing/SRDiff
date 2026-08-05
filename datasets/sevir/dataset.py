"""Map-style loader for aligned SEVIR satellite and VIL sequences.

Adapted from Amazon Science's Earthformer SEVIR loader under Apache-2.0.
See THIRD_PARTY_NOTICES.md.
"""

import os
from typing import Union, Dict, Sequence
import numpy as np
import datetime
import pandas as pd
import h5py
import torch
from pathlib import Path
from torch.nn.functional import avg_pool2d
from einops import rearrange
from torch.utils.data import Dataset

default_dataset_sevir_dir = os.environ.get("SEVIR_DATA_DIR", "work_dirs/data/sevir")

# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['ir069', 'ir107', 'lght', 'vil']
LIGHTNING_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
SEVIR_DATA_SHAPE = {'lght': (48, 48)}
PREPROCESS_SCALE_DDIM = {
    'ir069': 1.0 / 1174.68,
    'ir107': 1.0 / 2562.43,
    'vil': 1.0 / 255.0,
    'lght': 1.0 / 0.60517,
}
PREPROCESS_OFFSET_DDIM = {
    'ir069': 3683.58,
    'ir107': 1552.80,
    'vil': 0.0,
    'lght': -0.02990,
}

# SEVIR dataset paths
SEVIR_CATALOG = os.path.join(default_dataset_sevir_dir, "CATALOG.csv")
SEVIR_DATA_DIR = os.path.join(default_dataset_sevir_dir, "data")
SEVIR_RAW_SEQ_LEN = 49


def change_layout(
    data: Union[np.ndarray, torch.Tensor],
    in_layout: str = 'NHWT',
    out_layout: str = 'NHWT',
    ret_contiguous: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Rearranges the layout of the data tensor.

    Args:
        data (np.ndarray or torch.Tensor): The input data.
        in_layout (str): Input layout string. Default is 'NHWT'.
        out_layout (str): Output layout string. Default is 'NHWT'.
        ret_contiguous (bool): Whether to return a contiguous array/tensor.

    Returns:
        np.ndarray or torch.Tensor: The rearranged data.
    """
    in_layout = " ".join(in_layout.replace("C", "1"))
    out_layout = " ".join(out_layout.replace("C", "1"))
    data = rearrange(data, f"{in_layout} -> {out_layout}")
    if ret_contiguous:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
        elif isinstance(data, torch.Tensor):
            data = data.contiguous()
        else:
            raise ValueError("Data must be a numpy array or a torch tensor.")
    return data

class SEVIRDataset(Dataset):
    """Load aligned modalities and slice each event into fixed-stride windows."""

    def __init__(
            self,
            data_types: Sequence[str] = None,
            seq_len: int = 49,
            raw_seq_len: int = 49,
            stride: int = 12,
            layout: str = 'NHWT',
            sevir_catalog: Union[str, pd.DataFrame] = None,
            sevir_data_dir: str = None,
            start_date: datetime.datetime = None,
            end_date: datetime.datetime = None,
            output_type=np.float32,
            preprocess: bool = True,
            rescale_method: str = 'ddim',
            downsample_dict: Dict[str, Sequence[int]] = None,
            verbose: bool = False,
            ret_contiguous: bool = False,
        ):
        super().__init__()
        if sevir_catalog is None:
            sevir_catalog = SEVIR_CATALOG
        if sevir_data_dir is None:
            sevir_data_dir = SEVIR_DATA_DIR
        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        if set(data_types) != set(SEVIR_DATA_TYPES):
            raise ValueError(
                "The released experiment requires ir069, ir107, lght, and vil"
            )

        self.lght_frame_times = LIGHTNING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_len = raw_seq_len
        if not 0 < seq_len <= self.raw_seq_len:
            raise ValueError(
                f"seq_len must be in [1, {raw_seq_len}], got {seq_len}"
            )
        self.seq_len = seq_len
        if stride < 1:
            raise ValueError("stride must be positive")
        self.stride = stride
        valid_layout = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
        if layout not in valid_layout:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {valid_layout}.')
        self.layout = layout
        self._samples = None
        self._hdf_files = {}
        self._hdf_pid = None
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(sevir_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        if rescale_method != "ddim":
            raise ValueError("The released SEVIR pipeline requires rescale_method='ddim'")
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.ret_contiguous = ret_contiguous

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc >= self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc < self.end_date]
        self.catalog = self.catalog[self.catalog.pct_missing == 0]

        self._compute_samples()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        # TODO: is it necessary to keep one of them instead of deleting them all
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply(lambda df: self._df_to_series(df,imgt) )

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != 'lght' else s.id
            d.update({f'{i}_filename': [s.file_name],
                      f'{i}_index': [idx]})

        return pd.DataFrame(d)

    def _get_hdf_file(self, filename):
        """Open an HDF5 file lazily in the process that reads the sample."""
        current_pid = os.getpid()
        if self._hdf_pid != current_pid:
            # Never reuse h5py handles inherited from a parent process.
            self._hdf_files = {}
            self._hdf_pid = current_pid
        if filename not in self._hdf_files:
            path = Path(self.sevir_data_dir) / filename
            if self.verbose:
                print("Opening HDF5 file for reading", path)
            self._hdf_files[filename] = h5py.File(path, "r")
        return self._hdf_files[filename]

    def close(self):
        """
        Closes all open file handles
        """
        for handle in self._hdf_files.values():
            handle.close()
        self._hdf_files = {}
        self._hdf_pid = None

    def __getstate__(self):
        """Exclude process-local HDF5 handles when DataLoader uses spawn."""
        state = self.__dict__.copy()
        state["_hdf_files"] = {}
        state["_hdf_pid"] = None
        return state

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        return int(self.num_seq_per_event * self.total_num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            Dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx = row[f'{t}_index']
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            hdf_file = self._get_hdf_file(fname)
            if t == 'lght':
                lght_data = hdf_file[idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
                # data['event_id'] = idx
            else:
                data_i = hdf_file[t][idx:idx + 1, :, :, t_slice]
            data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (*self.data_shape['lght'], len(self.lght_frame_times)) if t_slice.stop is None else (*self.data_shape['lght'], 1)
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(t >= self.lght_frame_times[t_slice.stop - 1],
                                        t < self.lght_frame_times[t_slice.stop])
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(t >= self.lght_frame_times[0], t < self.lght_frame_times[1])
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        n = np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]
        return n

    def __len__(self):
        return self.total_num_seq

    def _load_event(self, event_idx):
        data = self._read_data(self._samples.iloc[event_idx], {})
        return {key: data[key].astype(self.output_type) for key in self.data_types}

    def __getitem__(self, index):
        data_dict = self._idx_sample(index=index)
        if self.ret_contiguous:
            for key in data_dict:
                data_dict[key] = data_dict[key].contiguous()
        return data_dict

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout='NHWT', rescale='ddim'):
        """Apply the released channel normalization and requested layout."""
        if rescale != 'ddim':
            raise ValueError("The released SEVIR pipeline supports only rescale='ddim'")
        scale_dict = PREPROCESS_SCALE_DDIM
        offset_dict = PREPROCESS_OFFSET_DDIM
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif isinstance(data, torch.Tensor):
                    data = data.float()
                else:
                    raise TypeError
                data = change_layout(data=scale_dict[key] * (data + offset_dict[key]),
                                     in_layout='NHWT',
                                     out_layout=layout)
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale='ddim'):
        """Invert the released channel normalization."""
        if rescale != 'ddim':
            raise ValueError("The released SEVIR pipeline supports only rescale='ddim'")
        scale_dict = PREPROCESS_SCALE_DDIM
        offset_dict = PREPROCESS_OFFSET_DDIM
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data = data_dict[key]
            data = data.float() / scale_dict[key] - offset_dict[key]
            data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to torch.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, torch.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = torch.from_numpy(data)
                else:
                    raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
            else:   # key == "mask"
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(data_dict, data_types=None, factors_dict=None, layout='NHWT'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.array, torch.Tensor]]
        factors_dict:   Optional[Dict[str, Sequence[int]]]
            each element `factors` is a Sequence of int, representing (t_factor, h_factor, w_factor)

        Returns
        -------
        downsampled_data_dict:  Dict[str, torch.Tensor]
            Modify on a deep copy of data_dict instead of directly modifying the original data_dict
        """
        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataset.data_dict_to_tensor(
            data_dict=data_dict,
            data_types=data_types)    # make a copy

        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout(
                    data=downsampled_data_dict[key],
                    in_layout=layout,
                    out_layout='NTHW')
                # downsample t dimension
                t_slice = [slice(None, None), ] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                # downsample spatial dimensions
                downsampled_data_dict[key] = avg_pool2d(
                    input=downsampled_data_dict[key],
                    kernel_size=(factors[1], factors[2]))

                downsampled_data_dict[key] = change_layout(
                    data=downsampled_data_dict[key],
                    in_layout='NTHW',
                    out_layout=layout)

        return downsampled_data_dict

    def _idx_sample(self, index):
        event_idx = index // self.num_seq_per_event
        seq_idx = index % self.num_seq_per_event
        event = self._load_event(event_idx)
        start = seq_idx * self.stride
        seq_slice = slice(start, start + self.seq_len)
        ret_dict = {
            key: event[key][..., seq_slice]
            for key in self.data_types
        }
        ret_dict = self.data_dict_to_tensor(ret_dict, data_types=self.data_types)

        if self.preprocess:
            ret_dict = self.preprocess_data_dict(
                data_dict=ret_dict,
                data_types=self.data_types,
                layout=self.layout,
                rescale=self.rescale_method,
            )
        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(
                data_dict=ret_dict,
                data_types=self.data_types,
                factors_dict=self.downsample_dict,
                layout=self.layout,
            )
        return ret_dict
