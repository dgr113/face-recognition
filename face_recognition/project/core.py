# coding: utf-8
from math import factorial
from time import sleep

import cv2
import json
import h5py
import logging
import asyncio
import numpy as np
import tensorflow as tf
from pathlib import Path
from itertools import starmap, repeat
from functools import partial
from sys import stderr, stdout
from operator import itemgetter
from collections import defaultdict, deque
from concurrent.futures.process import ProcessPoolExecutor
from typing import Union, Tuple, Mapping, Sequence, Any, Iterable, List, Dict
from jsonschema import Draft4Validator
from keras.layers import deserialize
from keras.models import Model, load_model
from keras.utils import to_categorical
from more_itertools import first, always_iterable, spy, collapse

tf.get_logger().setLevel(logging.ERROR)  ### Disable Tensorflow warning and other (non error) messages




class SystemUtils:
    """ Utilities to work system """

    @staticmethod
    def print_err(msg: Any, is_not_suppress: bool = True, ch: str = 'stderr', *args, **kwargs) -> None:
        """ Print Error into stderror/stdout

            :param msg: Message to output
            :param ch: Channel to output (stdout or stderr)
            :param is_not_suppress: Not suppress output
        """
        if is_not_suppress:
            ch = stdout if ch == 'stdout' else stderr
            print(msg, *args, file=ch, **kwargs)



class DataUtils:
    """ Utilities for working with data in different formats and structures """

    @staticmethod
    def dict_slice(d: Mapping, sliced_keys: Union[Union[int, str], Iterable[Union[int, str]]]) -> Mapping:
        """ Slice dict with keys """
        sliced_keys = tuple(always_iterable(sliced_keys))
        return { k: d[k] for k in d if k in sliced_keys }


    @staticmethod
    def get_enumerate_mapping(d: Mapping[str, List[dict]], field_key: str = 'data') -> Mapping[int, dict]:
        """ Get enumerate mapping (like Labels-Mapping) for Json struct """
        return { str(k): v for k, v in enumerate(CliUtils.mapping_flex_loader(d, {}).get(field_key, [])) }




class CliUtils:
    """ Helper utilities for CLI shells and validate input """

    @staticmethod
    def mapping_flex_loader(source: Union[Path, str, Mapping], default: Any = None, is_error_suppress: bool = False) -> Union[Mapping, Any]:
        """ Flexible get mapping from json file or Python Mapping

            :param source: Source file or Python Mapping object
            :param default: Default value for return on error
            :param is_error_suppress: Suppress mode on ?
        """

        if isinstance(source, Mapping):
            result = source

        else:
            result = default
            path_or_json = Path(source)

            if path_or_json.is_file():
                f = open(source, 'r')
                if f.readable():
                    try: result = json.load(f)
                    except ValueError:
                        fo_err_msg = "Error: {} path exists but failed to open!".format(path_or_json)
                        if is_error_suppress: SystemUtils.print_err(fo_err_msg)
                        else: raise FileExistsError(fo_err_msg)
                    finally: f.close()
                else: SystemUtils.print_err("Error: {} path exists but not readable!".format(path_or_json))

            else:
                try: result = json.loads(path_or_json)
                except ValueError: SystemUtils.print_err("Error: {} is not a path or valid JSON string!".format(path_or_json))

        return result


    @staticmethod
    def validate_json(schema_name: str, json_path: Union[Path, str], *, schema_mapping: Mapping) -> Tuple[str, Union[Mapping, None]]:
        """ Validate json file by schema

            :param schema_name: Schema name for access for JSON schema Mapping from <schema_mapping>
            :param json_path: JSON string or path to JSON file
            :param schema_mapping: Schemas Mapping
        """

        result = schema_name, None
        schema = CliUtils.mapping_flex_loader(schema_mapping[schema_name], is_error_suppress=True)
        data = CliUtils.mapping_flex_loader(json_path, is_error_suppress=True)

        if schema and data:
            try:
                validator = Draft4Validator(schema)
                validator.validate(data)
            except Exception as err: SystemUtils.print_err(err)
            else: result = schema_name, data

        return result


    @staticmethod
    def chain_validate(
        validate_data: Mapping[str, Any],
        schema_mapping: Mapping[str, str],
        data_slice_keys: Union[int, str, Iterable[int], Iterable[str], None] = None

    ) -> Mapping[str, Any]:

        """ Chain validate json files with their schemas """

        if data_slice_keys:
            validate_data = DataUtils.dict_slice(validate_data, data_slice_keys)

        validate_results = dict(starmap(
            partial(CliUtils.validate_json, schema_mapping=schema_mapping),
            validate_data.items()
        ))

        return validate_results



class DrawUtils:

    @staticmethod
    def draw_face_area(img, coords: Tuple[int, int, int, int]) -> None:
        """ Draw face area

            :param img: Image
            :param coords: Rectangle coords [up_left_x, up_left_y, down_right_x, down_right_y]
        """
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)



class ImageUtils:

    @staticmethod
    def resize_frame(img: np.array, *, new_shape: Tuple[int, ...]) -> np.array:
        """ Resize image to fixed size """
        return cv2.resize(img, new_shape[:2], interpolation=cv2.INTER_AREA)


    @staticmethod
    def extract_frame_area(
        img: np.array,
        coords: Union[Tuple[int, int, int, int], None] = None,
        frame_reshape: Union[Tuple[int, ...], None] = None,
        as_grayscale: bool = False

    ) -> np.array:

        """ Extract frame area with additional options """

        frame_area = np.copy(img)

        if coords:
            x1, y1, x2, y2 = coords
            if y2 <= frame_area.shape[0] and x2 <= frame_area.shape[1]:
                frame_area = ImageUtils.resize_frame(img[y1:y2, x1:x2], new_shape=frame_reshape[:2]) if frame_reshape else img[y1:y2, x1:x2]
            else:
                SystemUtils.print_err('Error! Wrong area range to extract from image frame. Return original image...')

        if as_grayscale:
            frame_area = cv2.cvtColor(frame_area, cv2.COLOR_BGR2GRAY)

        return frame_area


    @staticmethod
    def get_faces_area(f_cascade: cv2.CascadeClassifier, colored_img: np.array, scaleFactor: float = 1.1) -> np.array:
        """ Return image with detected face into rectangle """

        frame_area = np.copy(colored_img)
        frame_area = cv2.cvtColor(frame_area, cv2.COLOR_BGR2GRAY)
        faces = f_cascade.detectMultiScale(frame_area, scaleFactor=scaleFactor, minNeighbors=5)
        results = np.array([ (x, y, (x+w), (y+h)) for (x, y, w, h) in faces ])
        return results



class ClassifyUtils:

    @staticmethod
    def prepare_model(d: Mapping, input_shape: Tuple[int, ...], n_classes: int) -> Mapping:
        """ Validate and prepare model if needed """
        dd = defaultdict(lambda: defaultdict(dict), **d)
        dd['config']['layers'][0]['batch_input_shape'] = [None, *input_shape]
        dd['config']['layers'][-1]['units'] = n_classes
        return dict(dd)


    @staticmethod
    def create_model(config_path: str, input_shape: Tuple[int, ...], n_classes: int) -> Model:
        """ Create classify model """
        with open(config_path,  'r') as f:
            model_config = ClassifyUtils.prepare_model(json.load(f), input_shape, n_classes)
            model = deserialize(model_config)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            return model


    @staticmethod
    def fit_model(model, X_train, Y_train, X_test=None, Y_test=None, batch_size: int = 256, epochs: int = 100, verbose_mode: bool = True) -> Model:
        """ Train and return fitted model """
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose_mode)
        if X_test and Y_test:
            model.evaluate(X_test, Y_test)

        return model



class ProcessingUtils:

    @staticmethod
    def get_meta_from_hdf5(path: str, meta_field: str = 'metadata') -> Union[Dict[str, dict], None]:
        """ Get metainfo from group saved in HDF5 database """
        with h5py.File(path, 'r') as hdf:
            return {
                group_key: json.loads(hdf["{0}/{1}".format(group_key, meta_field)][()])
                for group_key in hdf.keys()
            }


    @staticmethod
    def _labels_prepare(labels: Union[Mapping[str, int], Sequence[str]]) -> Mapping[str, int]:
        """ Prepare labels """
        if isinstance(labels, Sequence):
            labels = { k: i for k, i in enumerate(set(str(x).casefold() for x in labels)) }

        return labels


    @staticmethod
    def face_predict(
        model,
        label_mapping: Mapping,
        frame: np.array,
        extracted_face: np.array,
        face_coords: Tuple[int, int, int, int]

    ) -> None:

        """ Predict extracted face and draw label with this name """

        ### Predict label ###
        predicted_code = model.predict_classes( np.array([extracted_face]) )
        predicted_meta = label_mapping.get(str(first(predicted_code, None)), {})

        ### Draw face area ###
        DrawUtils.draw_face_area(frame, face_coords)

        ### Draw person info ###
        for i, (k, v) in enumerate(predicted_meta.items()):
            msg = "{0}: {1}".format(k, v)
            x, y, *_ = face_coords[2:4]
            text = cv2.putText(frame, msg, (x, y-i*20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', text)


    @staticmethod
    def save_hdf5(
        chunked_data,
        person: Mapping,
        image_shape=(3, 3),
        storage_path: str = './data.hdf5',
        images_ds_name: str = 'X',
        labels_ds_name: str = 'y',
        metadata_ds_name: str = 'metadata'

    ) -> None:

        """ Chunked and incremental saving data in HDF5 """

        images, labels = zip(*chunked_data)

        with h5py.File(storage_path, 'a') as hdf:
            new_X_len, new_y_len = map(len, [images, labels])
            [group_label], group_labels = spy(labels)
            person_group_name = str(group_label)

            if person_group_name not in hdf:
                g = hdf.create_group(person_group_name)
                X = g.create_dataset(images_ds_name, (0, *image_shape), chunks=(*image_shape, 1), maxshape=(None, *image_shape))
                y = g.create_dataset(labels_ds_name, (0,), chunks=(1,), maxshape=(None,))
                g.create_dataset(metadata_ds_name, data=json.dumps(person))
            else:
                X = hdf[person_group_name][images_ds_name]
                y = hdf[person_group_name][labels_ds_name]

            X.resize(X.shape[0]+new_X_len, axis=0)
            y.resize(y.shape[0]+new_y_len, axis=0)
            X[-new_X_len:] = images
            y[-new_y_len:] = labels
            hdf.flush()

            print('Hello from blocking task!')


    @staticmethod
    async def wrap_task(pool, q, person, camera_frame_shape, save_data_path):
        loop = asyncio.get_running_loop()
        chunked_data = await q.get()
        return loop.run_in_executor(pool, ProcessingUtils.save_hdf5, chunked_data, person, camera_frame_shape, save_data_path)


    @staticmethod
    async def stage_learn_processing(
        camera_id: int,
        camera_close_key: str,
        haar_face_cascade: cv2.CascadeClassifier,
        camera_frame_shape: Tuple[int, ...],
        curr_label: int,
        person: Mapping,
        save_data_path: str,
        max_buffer_len: int = 1E2,
        max_queue_size: int = 1E3

    ) -> None:

        """ One Person processing """

        cam = cv2.VideoCapture(camera_id)

        with ProcessPoolExecutor(1) as pool:
            tasks = []
            chunked_data = deque()
            q = asyncio.Queue(maxsize=max_queue_size)
            try:
                shift = 0
                while True:
                    _, frame = cam.read()
                    faces_coords = ImageUtils.get_faces_area(haar_face_cascade, frame)
                    if faces_coords.any():
                        first_face_coords = itemgetter(0, 1, 2, 3)(faces_coords[0])  ### Use first face only
                        DrawUtils.draw_face_area(frame, first_face_coords)
                        frame_area = ImageUtils.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape)

                        if shift > max_buffer_len:
                            q.put_nowait(chunked_data.copy())
                            task = await asyncio.create_task(ProcessingUtils.wrap_task(pool, q, person, camera_frame_shape, save_data_path))
                            tasks.append(task)
                            chunked_data.clear()
                            shift = 0
                        else:
                            chunked_data.append([frame_area, curr_label])
                            shift += 1

                    if cv2.waitKey(30) == ord(camera_close_key):
                        # task = asyncio.create_task(ProcessingUtils.save_task(loop, pool, q, person, camera_frame_shape, save_data_path))
                        # tasks.append(task)
                        break
                    else:
                        cv2.imshow('frame', frame)

            finally:
                # results = await asyncio.gather(*tasks, return_exceptions=True)
                cv2.destroyAllWindows()
                cam.release()


    @staticmethod
    def start_learn(
        persons: Sequence[Mapping],
        camera: Mapping,
        model_config_path: str,
        haar_cascade_path: str,
        save_data_path: Union[str, None] = None

    ) -> Model:

        """ Learn model by faces from camera """

        camera_id = camera['camera_id']
        camera_frame_shape = tuple(camera['camera_frame_shape'])
        camera_close_key = camera['camera_close_key']
        n_classes = len(persons)

        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        model = ClassifyUtils.create_model(model_config_path, camera_frame_shape, n_classes)

        for label, person in enumerate(persons):
            inp_msg = "{0} {1}, are you ready? [Y/n]: ".format(*map(str.capitalize, always_iterable(itemgetter('first_name', 'last_name')(person))))
            inp_val = input(inp_msg)
            if inp_val.lower() == 'y':
                asyncio.run(ProcessingUtils.stage_learn_processing(camera_id, camera_close_key, haar_face_cascade, camera_frame_shape, label, person, save_data_path))

        # ### Learn models
        # images = np.array(tuple(collapse(images, levels=1)))
        # dummy_labels = to_categorical(tuple(collapse(labels, levels=1)))
        # fitted_model = ClassifyUtils.fit_model(model=model, X_train=images, Y_train=dummy_labels)
        # return fitted_model


    @staticmethod
    async def start_predict(persons_mapping: Mapping, camera: Mapping, model_path: str, haar_cascade_path: str) -> None:
        """ Recognition faces from camera """

        camera_id = camera['camera_id']
        camera_frame_shape = tuple(camera['camera_frame_shape'])
        camera_close_key = camera['camera_close_key']

        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        model = load_model(model_path)
        predict_func = partial(ProcessingUtils.face_predict, model, persons_mapping)

        cam = cv2.VideoCapture(camera_id)
        while True:
            _, frame = cam.read()
            cv2.imshow('frame', frame)

            faces_coords = ImageUtils.get_faces_area(haar_face_cascade, frame)
            if faces_coords.any():
                first_face_coords = itemgetter(0, 1, 2, 3)(faces_coords[0])  ### Use first face only
                extracted_face = ImageUtils.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape)
                predict_func(frame, extracted_face, first_face_coords)

            if cv2.waitKey(30) == ord(camera_close_key):
                break

        cam.release()
        cv2.destroyAllWindows()
