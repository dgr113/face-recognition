# coding: utf-8

import os
import sys
import asyncio
from pathlib import Path
from functools import partial
from itertools import chain
from operator import itemgetter
from argparse import ArgumentParser
from typing import Mapping, Union, Iterable, Any, Tuple, Sequence
from more_itertools import ilen

sys.path.append( os.path.abspath( os.path.join(os.path.dirname(__file__), '../..') ) )
from face_recognition.project.core import TrainsetCreation, SystemUtils, CliUtils, DataUtils, Processing, HDF5Utils
from face_recognition.project.schema import SCHEMA_MAPPING




def _get_persons_flexible(data: str, persons: Union[Mapping, str], data_field: str, persons_field: str) -> Mapping:
    """ Flexible get Persons Metadata """
    if data:
        persons = HDF5Utils.get_hdf5_data(data, data_field)
    else:
        persons = DataUtils.get_enumerate_mapping(CliUtils.mapping_flex_loader(persons), persons_field)

    return persons



def zip_dataset(data: Iterable[Mapping[str, Any]]) -> Tuple[Sequence[Any], Sequence[Any]]:
    X, y = map(tuple, map(chain.from_iterable, zip(*map(itemgetter('X', 'y'), data))))
    return X, y




def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['fit', 'predict'], help='Program mode - train or predict faces')
    parser.add_argument('-mc', '--model-save-path', type=str, default='../_instance/output/model.hdf5', help='Path to save classification model')
    parser.add_argument('-ms', '--model-config-path', type=str, default='../_instance/config/model_config.json', help='Path to json model configuration')
    parser.add_argument('-hc', '--haar-path', type=str, default='../_instance/config/haarcascade_frontalface_alt.xml', help='Path to describe Haarcascade file')
    parser.add_argument('-d', '--data-path', type=str, default='../_instance/output/data.hdf5', help='Path to save learning dataset')
    parser.add_argument('-c', '--camera', type=str, default='../_instance/config/camera.json', help='Camera json config')
    parser.add_argument('-l', '--persons', type=str, default='../_instance/config/persons.json', help='Labels json config')
    args = vars(parser.parse_args())


    if args['mode'] == 'fit':
        validate_results = CliUtils.chain_validate(args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])
        if all(validate_results.values()):
            # TrainsetCreation.create_datasets(validate_results['persons']['data'], validate_results['camera']['data'], args['haar_path'], args['data_path'])

            classes_count = ilen(HDF5Utils.get_hdf5_data(Path(args['data_path'])))
            step_per_epoch = ilen(HDF5Utils.hdf5_train_data_gen(args['data_path']))
            epoch = 10

            model = Processing.start_learn_test(
                generator=HDF5Utils.hdf5_train_data_gen(args['data_path'], n_classes=classes_count, is_infinite=True),
                model_config_path=args['model_config_path'],
                camera_frame_shape=validate_results['camera']['data']['camera_frame_shape'],
                n_classes=classes_count,
                epoch=epoch,
                step_per_epoch=step_per_epoch
            )
            model.save(args['model_save_path'])


    elif args['mode'] == 'predict':
        persons = _get_persons_flexible(args['data_path'], args['persons'], 'metadata', 'data')
        validate_args = {'persons': {'data': list(persons.values())}, 'camera': args['camera']}
        validate_results = CliUtils.chain_validate(validate_args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])

        if all(validate_results.values()):
            asyncio.run(Processing.start_predict(
                persons,
                validate_results['camera']['data'],
                args['model_save_path'],
                args['haar_path']
            ))

    else: SystemUtils.print_err("Validate Error!")





if __name__ == "__main__":
    main()
