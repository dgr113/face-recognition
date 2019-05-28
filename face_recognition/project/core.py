# coding: utf-8

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Union, Tuple, Mapping, Sequence, Any, Iterable, List, Dict, Hashable, Callable, Generator
from jsonschema import Draft4Validator
from keras.layers import deserialize
from keras.models import Model, load_model
from keras.utils import to_categorical
from more_itertools import first, always_iterable, spy, collapse

tf.get_logger().setLevel(logging.ERROR)  ### Disable Tensorflow warning and other (non error) messages




class SystemUtils:
    """ System utility """

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
    """ Utilities for general working with data in different formats and structures """

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
    """ Command line and validate tools """

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




class ImageWorking:

    @staticmethod
    def draw_face_area(img, coords: Tuple[int, int, int, int]) -> None:
        """ Draw face area

            :param img: Image
            :param coords: Rectangle coords [up_left_x, up_left_y, down_right_x, down_right_y]
        """
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)


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
                frame_area = ImageWorking.resize_frame(img[y1:y2, x1:x2], new_shape=frame_reshape[:2]) if frame_reshape else img[y1:y2, x1:x2]
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




class LearnModel:

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
            model_config = LearnModel.prepare_model(json.load(f), input_shape, n_classes)
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


    @staticmethod
    def fit_model_gen(model, generator, X_test=None, Y_test=None, epochs: int = 1, steps_per_epoch: int = 1, verbose_mode: bool = True) -> Model:
        """ Train and return fitted model """
        model.fit_generator(generator=generator, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose_mode)
        if X_test and Y_test:
            model.evaluate(X_test, Y_test)

        return model




class HDF5Utils:
    """ HDF5 data storage utilities """

    @staticmethod
    def json_data_handle(field_name: str, field_data: Any, json_fields: Union[Sequence[Hashable], None] = None) -> Any:
        """ Handle json data from HDF5 datasets """
        return json.loads(field_data) if field_name in (json_fields or []) else field_data


    @staticmethod
    def get_hdf5_data(data_path: Union[str, Path], fields: Union[Sequence[Hashable], None] = None, *, json_fields: Union[Sequence[Hashable], None] = None) -> Union[Dict[str, dict], None]:
        """ Get data from datasets(fields) or groups saved in HDF5 database
            If <fields> param is not set, only group names are returned

            :param data_path: Path to HDF5 data storage
            :param fields: Required dataset names for return
            :param json_fields: Fields that are JSON-encoded strings
        """

        fields = fields or []
        json_fields = json_fields or []
        json_handle_func = partial(HDF5Utils.json_data_handle, json_fields=json_fields)

        with h5py.File(data_path, 'r') as hdf:
            return {
                group_key: {
                    field: json_handle_func(field, hdf["{0}/{1}".format(group_key, field)][()])
                    for field in fields
                }
                for group_key in hdf.keys()
            }


    @staticmethod
    def hdf5_train_data_gen(
        data_path: str,
        X_field: str = 'X',
        y_field: str = 'y',
        batch_size: int = 200,
        n_classes: Union[int, None] = None,
        is_infinite: bool = False

    ) -> Union[Dict[str, dict], None]:

        """ Batches generator of X, y chunks """

        with h5py.File(data_path, 'r') as hdf:
            while True:
                for group_key in hdf.keys():
                    start_pos, end_pos = 0, batch_size
                    while True:
                        group_data = hdf[group_key]
                        X = group_data[X_field][start_pos:end_pos]
                        y = group_data[y_field][start_pos:end_pos]
                        if X.size and y.size:
                            yield X, to_categorical(y, num_classes=n_classes)
                            start_pos = batch_size
                            end_pos += batch_size
                        else: break

                if not is_infinite:
                    break


    @staticmethod
    def save_hdf5_train_images(
        chunked_data,
        metadata: Mapping,
        image_shape,
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
                g.create_dataset(metadata_ds_name, data=json.dumps(metadata))
            else:
                X = hdf[person_group_name][images_ds_name]
                y = hdf[person_group_name][labels_ds_name]

            X.resize(X.shape[0]+new_X_len, axis=0)
            y.resize(y.shape[0]+new_y_len, axis=0)
            X[-new_X_len:] = images
            y[-new_y_len:] = labels
            hdf.flush()


    @staticmethod
    async def save_hdf5_train_images_async(
        save_data_path: Union[str, Path],
        q: asyncio.Queue,
        image_frame_shape,
        metadata: Mapping,
        pool: Union[ThreadPoolExecutor, ProcessPoolExecutor] = None

    ) -> Generator:

        """ Wrap <run_in_executor> future into async coroutine """

        loop = asyncio.get_running_loop()
        chunked_data = await q.get()
        return loop.run_in_executor(pool, HDF5Utils.save_hdf5_train_images, chunked_data, metadata, image_frame_shape, save_data_path)




class TrainsetCreation:
    """ Utilities to image cropping, learn-datasets creation and saving their """

    @staticmethod
    async def _create_stage_dataset(
        camera_id: int,
        camera_close_key: str,
        haar_face_cascade: cv2.CascadeClassifier,
        camera_frame_shape: Tuple[int, ...],
        curr_label: int,
        person: Mapping,
        save_data_path: str,
        max_buffer_len: int = 1E1,
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
                    faces_coords = ImageWorking.get_faces_area(haar_face_cascade, frame)
                    if faces_coords.any():
                        first_face_coords = itemgetter(0, 1, 2, 3)(faces_coords[0])  ### Use first face only
                        ImageWorking.draw_face_area(frame, first_face_coords)
                        frame_area = ImageWorking.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape)

                        if shift > max_buffer_len:
                            q.put_nowait(chunked_data.copy())
                            task = await asyncio.create_task(
                                HDF5Utils.save_hdf5_train_images_async(save_data_path, q, camera_frame_shape, person, pool)
                            )

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
    def create_datasets(persons: Sequence[Mapping], camera: Mapping, haar_cascade_path: str, save_data_path: Union[str, None] = None) -> None:
        """ Learn model by faces from camera """

        camera_id = camera['camera_id']
        camera_frame_shape = tuple(camera['camera_frame_shape'])
        camera_close_key = camera['camera_close_key']
        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)

        for label, person in enumerate(persons):
            inp_msg = "{0} {1}, are you ready? [Y/n]: ".format(*map(str.capitalize, always_iterable(itemgetter('first_name', 'last_name')(person))))
            inp_val = input(inp_msg)
            if inp_val.lower() == 'y':
                asyncio.run(TrainsetCreation._create_stage_dataset(camera_id, camera_close_key, haar_face_cascade, camera_frame_shape, label, person, save_data_path))




class Processing:
    """ Model learning and recognition processing """

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
        ImageWorking.draw_face_area(frame, face_coords)

        ### Draw person info ###
        for i, (k, v) in enumerate(predicted_meta.items()):
            msg = "{0}: {1}".format(k, v)
            x, y, *_ = face_coords[2:4]
            text = cv2.putText(frame, msg, (x, y-i*20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', text)


    @staticmethod
    def start_learn(X, y, model_config_path, camera_frame_shape, n_classes):
        """ Fit learn model from training dataset """
        dummy_labels = to_categorical(y)
        model = LearnModel.create_model(model_config_path, camera_frame_shape, n_classes)
        return LearnModel.fit_model(model=model, X_train=X, Y_train=dummy_labels)


    @staticmethod
    def start_learn_test(generator, model_config_path: Union[str, Path], camera_frame_shape, n_classes, epoch: int = 1, step_per_epoch: int = 1):
        """ Fit learn model from training dataset """
        model = LearnModel.create_model(model_config_path, camera_frame_shape, n_classes)
        return LearnModel.fit_model_gen(model=model, generator=generator, epochs=epoch, steps_per_epoch=step_per_epoch)


    @staticmethod
    async def start_predict(persons_mapping: Mapping, camera: Mapping, model_path: str, haar_cascade_path: str) -> None:
        """ Recognition faces from camera """

        camera_id = camera['camera_id']
        camera_frame_shape = tuple(camera['camera_frame_shape'])
        camera_close_key = camera['camera_close_key']

        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        model = load_model(model_path)
        predict_func = partial(Processing.face_predict, model, persons_mapping)

        cam = cv2.VideoCapture(camera_id)
        while True:
            _, frame = cam.read()
            cv2.imshow('frame', frame)

            faces_coords = ImageWorking.get_faces_area(haar_face_cascade, frame)
            if faces_coords.any():
                first_face_coords = itemgetter(0, 1, 2, 3)(faces_coords[0])  ### Use first face only
                extracted_face = ImageWorking.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape)
                predict_func(frame, extracted_face, first_face_coords)

            if cv2.waitKey(30) == ord(camera_close_key):
                break

        cam.release()
        cv2.destroyAllWindows()
