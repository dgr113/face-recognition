# coding: utf-8

import os
import sys
import asyncio
from argparse import ArgumentParser
from typing import Mapping, Union

sys.path.append( os.path.abspath( os.path.join(os.path.dirname(__file__), '../..') ) )
from face_recognition.project.core import ProcessingUtils, SystemUtils, CliUtils, DataUtils
from face_recognition.project.schema import SCHEMA_MAPPING




def _get_persons_flexible(data: str, persons: Union[Mapping, str], data_field: str, persons_field: str) -> Mapping:
    """ Flexible get Persons Metadata """
    if data:
        persons = ProcessingUtils.get_meta_from_hdf5(data, data_field)
    else:
        persons = DataUtils.get_enumerate_mapping(CliUtils.mapping_flex_loader(persons), persons_field)

    return persons




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
            model = ProcessingUtils.start_learn(
                validate_results['persons']['data'],
                validate_results['camera']['data'],
                args['model_config_path'],
                args['haar_path'],
                args['data_path']
            )
            model.save(args['model_save_path'])


    elif args['mode'] == 'predict':
        persons = _get_persons_flexible(args['data_path'], args['persons'], 'metadata', 'data')
        validate_args = {'persons': {'data': list(persons.values())}, 'camera': args['camera']}
        validate_results = CliUtils.chain_validate(validate_args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])

        if all(validate_results.values()):
            asyncio.run(ProcessingUtils.start_predict(
                persons,
                validate_results['camera']['data'],
                args['model_save_path'],
                args['haar_path']
            ))

    else: SystemUtils.print_err("Validate Error!")





if __name__ == "__main__":
    main()
