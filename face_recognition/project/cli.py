# coding: utf-8

import os
import sys
import asyncio
from pprint import pprint
from argparse import ArgumentParser

sys.path.append( os.path.abspath( os.path.join(os.path.dirname(__file__), '../..') ) )
from face_recognition.project.core import TrainsetCreation, SystemUtils, CliUtils, PredictModel, LearnModel, ProductSequence
from face_recognition.project.schema import SCHEMA_MAPPING




def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['cutting', 'fit', 'predict', 'schema'], help='Program mode - create dataset from camera, fit model, predict faces or data input schema return')
    parser.add_argument('-mc', '--model-save-path', type=str, default='../_instance/output/model.hdf5', help='Path to save classification model')
    parser.add_argument('-ms', '--model-config-path', type=str, default='../_instance/config/model_config.json', help='Path to json model configuration')
    parser.add_argument('-hc', '--haar-path', type=str, default='../_instance/config/haarcascade_frontalface_alt.xml', help='Path to describe Haarcascade file')
    parser.add_argument('-d', '--data-path', type=str, default='../_instance/output/data.hdf5', help='Path to save learning dataset')
    parser.add_argument('-c', '--camera', type=str, default='../_instance/config/camera_config.json', help='Camera json config')
    parser.add_argument('-p', '--persons', type=str, default='../_instance/config/persons.json', help='Labels json config')
    parser.add_argument('-l', '--learn', type=str, default='../_instance/config/learn_config.json', help='Learn process config')
    args = vars(parser.parse_args())


    if args['mode'] == 'cutting':
        validate_results = CliUtils.chain_validate(args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])
        if all(validate_results.values()):
            TrainsetCreation.create_datasets(
                validate_results['persons'],
                validate_results['camera'],
                args['haar_path'],
                args['data_path'],
                as_grayscale=True
            )
        else: SystemUtils.print_err("Validate Error!")


    if args['mode'] == 'fit':
        validate_results = CliUtils.chain_validate(args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons', 'learn'])
        if all(validate_results.values()):
            model = LearnModel.start_learn(
                ProductSequence(args['data_path']),
                args['model_config_path'],
                validate_results['camera'],
                validate_results['learn'],
                n_classes=2
            )
            model.save(args['model_save_path'])
        else: SystemUtils.print_err("Validate Error!")


    elif args['mode'] == 'predict':
        args.update({'persons': CliUtils.get_meta_flexible(args['data_path'] or args['persons'], 'metadata')})
        validate_results = CliUtils.chain_validate(args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])

        if all(validate_results.values()):
            asyncio.run(PredictModel.start_predict(
                validate_results['persons'],
                validate_results['camera'],
                args['model_save_path'],
                args['haar_path'],
                as_grayscale=True
            ))
        else: SystemUtils.print_err("Validate Error!")


    elif args['mode'] == 'schema':
        pprint(SCHEMA_MAPPING)





if __name__ == "__main__":
    main()
