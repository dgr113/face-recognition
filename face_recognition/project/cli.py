# coding: utf-8

import os
import sys
import asyncio
from argparse import ArgumentParser

sys.path.append( os.path.abspath( os.path.join(os.path.dirname(__file__), '../..') ) )
from face_recognition.project.core import TrainsetCreation, SystemUtils, CliUtils, PredictModel, HDF5Utils, LearnModel
from face_recognition.project.schema import SCHEMA_MAPPING




def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['cutting', 'fit', 'predict'], help='Program mode - create dataset from camera, fit model or predict faces')
    parser.add_argument('-mc', '--model-save-path', type=str, default='../_instance/output/model.hdf5', help='Path to save classification model')
    parser.add_argument('-ms', '--model-config-path', type=str, default='../_instance/config/model_config.json', help='Path to json model configuration')
    parser.add_argument('-hc', '--haar-path', type=str, default='../_instance/config/haarcascade_frontalface_alt.xml', help='Path to describe Haarcascade file')
    parser.add_argument('-d', '--data-path', type=str, default='../_instance/output/data.hdf5', help='Path to save learning dataset')
    parser.add_argument('-c', '--camera', type=str, default='../_instance/config/camera.json', help='Camera json config')
    parser.add_argument('-l', '--persons', type=str, default='../_instance/config/persons.json', help='Labels json config')
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


    if args['mode'] == 'fit':
        validate_results = CliUtils.chain_validate(args, schema_mapping=SCHEMA_MAPPING, data_slice_keys=['camera', 'persons'])
        if all(validate_results.values()):
            step_per_epoch = HDF5Utils.calc_hdf5_batch_count(args['data_path'], 'y')
            epoch = 20

            train_set = tuple(HDF5Utils.hdf5_train_data_gen(args['data_path'], is_infinite=False))
            model = LearnModel.start_learn(
                train_set,
                model_config_path=args['model_config_path'],
                camera_frame_shape=validate_results['camera']['camera_frame_shape'],
                n_classes=2,
                epoch=epoch,
                step_per_epoch=step_per_epoch
            )
            model.save(args['model_save_path'])


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

    else:
        SystemUtils.print_err("Validate Error!")





if __name__ == "__main__":
    main()
