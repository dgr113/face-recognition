# coding: utf-8

import cv2
import json
import h5py
import numpy as np
from functools import partial
from pathlib import Path
from typing import Union, Sequence, Hashable, Dict, Any
from more_itertools import always_iterable




def write_image(save_path: str, img: np.array) -> None:
    """ Write image from numpy array """
    cv2.imwrite(save_path, img)



def _json_data_handle(field_name: str, field_data: Any, json_fields: Union[Sequence[Hashable], None] = None) -> Any:
    """ Handle json data from HDF5 datasets """
    return json.loads(field_data) if field_name in (json_fields or []) else field_data



def get_hdf5_data(data_path: Union[str, Path], fields: Union[Sequence[Hashable], None] = None, *, json_fields: Union[Sequence[Hashable], None] = None) -> Union[Dict[str, dict], None]:
    """ Get data from datasets(fields) or groups saved in HDF5 database
        If <fields> param is not set, only group names are returned

        :param data_path: Path to HDF5 data storage
        :param fields: Required dataset names for return
        :param json_fields: Fields that are JSON-encoded strings
    """

    fields = tuple(always_iterable(fields or []))
    json_fields = tuple(always_iterable(json_fields or []))
    json_handle_func = partial(_json_data_handle, json_fields=json_fields)

    with h5py.File(data_path, 'r') as hdf:
        return {
            group_key: {
                field: json_handle_func(field, hdf["{0}/{1}".format(group_key, field)][()])
                for field in fields
            }
            for group_key in hdf.keys()
        }





def main():
    images = get_hdf5_data('../../_instance/output/data.hdf5', 'X')
    for i in range(2):
        for j, img in enumerate(images[str(i)]['X']):
            img_name = "{0}{1}/{2}{3}".format("../../_instance/images/", str(i), str(j), ".jpeg")
            write_image(img_name, img)



if __name__ == "__main__":
    main()
