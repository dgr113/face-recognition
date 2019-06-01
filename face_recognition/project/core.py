# coding: utf-8
# python3.7

import cv2
import json
import h5py
import logging
import asyncio
import numpy as np
import tensorflow as tf
from pathlib import Path
from itertools import starmap
from functools import partial
from sys import stderr, stdout
from operator import itemgetter, truth
from collections import defaultdict, deque
from dataclasses import dataclass, astuple, InitVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Union, Mapping, Generator, Sequence, Tuple
from jsonschema import Draft4Validator
from keras.layers import deserialize
from keras.models import Model, load_model
from keras.utils import Sequence as KerasSequence, to_categorical
from more_itertools import first, always_iterable, spy, collapse
from face_recognition.project.type_hints import CHUNKED_DATA_TYPE, ONE_MORE_KEYS_TYPE, UNIVERSAL_SOURCE_TYPE, \
    POINT_COORDS_TYPE
from face_recognition.project.type_hints import KEYS_OR_NONE_TYPE, MODEL_CONFIG_TYPE, COLOR_TYPE, JSON_DATA_TYPES
from face_recognition.project.type_hints import TRAIN_DATA_GEN_TYPE, TRAIN_DATA_TYPE, TRAIN_LABELS_TYPE
from face_recognition.project.type_hints import UNIVERSAL_PATH_TYPE, UNIVERSAL_CONFIG_TYPE, HDF5_DATA_TYPE, HDF5_GROUPED_DATA_TYPE
from face_recognition.project.type_hints import VALIDATE_RESUTS_TYPE, COORDS_TYPE, FRAME_SHAPE_TYPE

tf.get_logger().setLevel(logging.ERROR)  ### Disable Tensorflow warning and other (non error) messages




@dataclass(frozen=True)
class RectArea:
    """ Rectangle area description """

    x1: int  # Left up <X> coordinate
    y1: int  # Left up <Y> coordinate
    x2: int  # Right down <X> coordinate
    y2: int  # Right down <Y> coordinate

    def to_points(self) -> Tuple[POINT_COORDS_TYPE, POINT_COORDS_TYPE]:
        return (self.x1, self.y1), (self.x2, self.y2)




@dataclass
class ProductSequence(KerasSequence):
    """ Keras model fit train batch generator """

    data_path: UNIVERSAL_PATH_TYPE  # Path to HDF5 storage
    X_field: str = 'X'  # Name of dataset with train data
    y_field: str = 'y'  # Name of dataset with train labels
    batch_size: InitVar[int] = 30  # Data chunk size on the one Keras model fit iteration

    def __post_init__(self, batch_size: int):
        with h5py.File(self.data_path, 'r') as hdf:
            self.classes_count = len(hdf.keys())
            self.batch_shift = batch_size // self.classes_count
            self.step_per_epoch = min(hdf[group_key][self.y_field].shape[0] for group_key in hdf.keys()) // self.batch_shift


    def __len__(self) -> int:
        """ Returns number of step(chunks) in epoch """
        return self.step_per_epoch


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a training data chunk """

        with h5py.File(self.data_path, 'r') as hdf:
            start_pos, end_pos = self.batch_shift * idx, self.batch_shift * (idx + 1)
            X, y = [], []
            for group in hdf.keys():
                group_data = hdf[group]
                X.extend( group_data[self.X_field][start_pos:end_pos] )
                y.extend( group_data[self.y_field][start_pos:end_pos] )

            return np.array(X), to_categorical(y, self.classes_count)




class SystemUtils:
    """ System utility """

    @staticmethod
    def print_err(msg: str, is_not_suppress: bool = True, ch: str = 'stderr', *args, **kwargs) -> None:
        """ Print Error into selected channel

            :param msg: Message to output
            :param ch: Channel to output (stdout or stderr)
            :param is_not_suppress: Output is not suppressed
        """
        if is_not_suppress:
            ch = stdout if ch == 'stdout' else stderr
            print(msg, *args, file=ch, **kwargs)




class DataUtils:
    """ Utilities for general working with data in different formats and structures """

    @staticmethod
    def dict_slice(d: Mapping, sliced_keys: ONE_MORE_KEYS_TYPE) -> dict:
        """ Slice dict by keys

            :param d: Mapping for slicing dy selective keys
            :param sliced_keys: Keys for slicing
        """
        return { k: d[k] for k in d if k in tuple(always_iterable(sliced_keys)) }




class CliUtils:
    """ Command line and validate utils """

    @staticmethod
    def get_meta_flexible(source: UNIVERSAL_SOURCE_TYPE, source_field: str = 'metadata') -> Mapping:
        """ Flexible get metadata dataset from HDF5 storage """
        if isinstance(source, (str, Path)) and str(source).endswith('hdf5'):
            metadata = HDF5Utils.get_hdf5_data(source, fields=source_field, json_fields=source_field)
            source = { k: v.get(source_field, {}) for k, v in metadata.items() }
        return source


    @staticmethod
    def mapping_flex_loader(source: UNIVERSAL_SOURCE_TYPE, default: JSON_DATA_TYPES = None, is_error_suppress: bool = False) -> UNIVERSAL_CONFIG_TYPE:
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
    def validate_json(schema_name: str, json_path: UNIVERSAL_PATH_TYPE, *, schema_mapping: UNIVERSAL_CONFIG_TYPE) -> VALIDATE_RESUTS_TYPE:
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
            except Exception as err: SystemUtils.print_err(str(err))
            else: result = schema_name, data

        return result


    @staticmethod
    def chain_validate(
        data: UNIVERSAL_CONFIG_TYPE,
        schema_mapping: UNIVERSAL_CONFIG_TYPE,
        data_slice_keys: Union[ONE_MORE_KEYS_TYPE, None] = None

    ) -> Mapping[str, UNIVERSAL_CONFIG_TYPE]:

        """ Chain validate json files by schemas description

            :param data: User input data for validation
            :param schema_mapping: Json-schemes description mapping
            :param data_slice_keys: Selective validation
        """

        if data_slice_keys:
            data = DataUtils.dict_slice(data, data_slice_keys)

        validate_results = dict(starmap(
            partial(CliUtils.validate_json, schema_mapping=schema_mapping),
            data.items()
        ))

        return validate_results




class ImageWorking:
    """ Utilities for working with images as a Numpy arrays """

    @staticmethod
    def _resize_frame(img: np.ndarray, *, new_shape: FRAME_SHAPE_TYPE, interpolation_code: int = cv2.INTER_AREA) -> np.ndarray:
        """ Resize image to fixed size

            :param img: Image as a numpy array
            :param new_shape: Shape for resizing image
        """
        return cv2.resize(img, new_shape[:2], interpolation=interpolation_code)


    @staticmethod
    def draw_rect_area(img: np.ndarray, coords: RectArea, color: COLOR_TYPE = (0, 255, 0)) -> None:
        """ Draw rectangle area

            :param img: Image as a numpy array
            :param coords: Rectangle coords [up_left_x, up_left_y, down_right_x, down_right_y]
            :param color: Color RGB code
        """
        cv2.rectangle(img, *coords.to_points(), color, 2)


    @staticmethod
    def extract_frame_area(
        img: np.ndarray,
        coords: Union[RectArea, None] = None,
        frame_reshape: Union[FRAME_SHAPE_TYPE, None] = None,
        *, as_grayscale: bool = False

    ) -> np.ndarray:

        """ Extract frame area with additional options

            :param img: Image as a numpy array
            :param coords: Rectangle coords [up_left_x, up_left_y, down_right_x, down_right_y]
            :param frame_reshape: Shape for resizing image
            :param as_grayscale: Extract frame as grayscale
        """

        frame_area = np.copy(img)

        if coords:
            x1, y1, x2, y2 = astuple(coords)
            if y2 <= frame_area.shape[0] and x2 <= frame_area.shape[1]:
                frame_area = ImageWorking._resize_frame(img[y1:y2, x1:x2], new_shape=frame_reshape[:2]) if frame_reshape else img[y1:y2, x1:x2]
            else:
                SystemUtils.print_err('Error! Wrong area range to extract from image frame. Return original image...')

        if as_grayscale:
            frame_area = cv2.cvtColor(frame_area, cv2.COLOR_BGR2GRAY)
            frame_area = frame_area[:, :, np.newaxis]  ### Add an additional axis to indicate an empty color channel (for Tensorflow backend)

        return frame_area


    @staticmethod
    def get_faces_area(f_cascade: cv2.CascadeClassifier, img: np.ndarray, scaleFactor: float = 1.1) -> Sequence[COORDS_TYPE]:
        """ Return extracted face coordinates

            :param f_cascade: Cascade classifier for extract face from image frame
            :param img: Image frame
            :param scaleFactor: Scale factor for extract faces
        """
        frame_area = np.copy(img)
        frame_area = cv2.cvtColor(frame_area, cv2.COLOR_BGR2GRAY)
        faces = f_cascade.detectMultiScale(frame_area, scaleFactor=scaleFactor, minNeighbors=5)
        results = [ (x, y, (x+w), (y+h)) for (x, y, w, h) in faces ]
        return results




class HDF5Utils:
    """ HDF5 data storage utilities """

    @staticmethod
    def _json_data_handle(field_name: str, field_data: HDF5_DATA_TYPE, json_fields: Union[str, None] = None) -> HDF5_DATA_TYPE:
        """ Handle json data from HDF5 datasets

            :param field_name: Field (dataset) name
            :param field_data: Field (dataset) data
            :param json_fields: Field (dataset) names are interpreted as Json
        """
        return json.loads(field_data) if field_name in (json_fields or []) else field_data


    @staticmethod
    def get_hdf5_data(
        data_path: UNIVERSAL_PATH_TYPE,
        fields: KEYS_OR_NONE_TYPE = None,
        *, json_fields: KEYS_OR_NONE_TYPE = None

    ) -> Union[HDF5_GROUPED_DATA_TYPE, None]:

        """ Get data from datasets(fields) or groups saved in HDF5 database
            If <fields> param is not set, only group names are returned

            :param data_path: Path to HDF5 data storage
            :param fields: Required dataset names for return
            :param json_fields: Fields that are JSON-encoded strings
        """

        fields = set(always_iterable(fields or []))
        json_fields = set(always_iterable(json_fields or []))
        json_handle_func = partial(HDF5Utils._json_data_handle, json_fields=json_fields)

        with h5py.File(data_path, 'r') as hdf:
            return {
                group_key: { field: json_handle_func(field, hdf[group_key][field][()]) for field in fields }
                for group_key in hdf.keys()
            }


    @staticmethod
    def save_hdf5_train_images(
        chunked_data: CHUNKED_DATA_TYPE,
        metadata: UNIVERSAL_CONFIG_TYPE,
        image_shape: FRAME_SHAPE_TYPE,
        data_path: UNIVERSAL_PATH_TYPE = './data.hdf5',
        images_ds_name: str = 'X',
        labels_ds_name: str = 'y',
        metadata_ds_name: str = 'metadata'

    ) -> None:

        """ Chunked and incremental saving data in HDF5

            :param chunked_data:
            :param metadata:
            :param image_shape:
            :param data_path:
            :param images_ds_name:
            :param labels_ds_name:
            :param metadata_ds_name:
        """

        image_shape = tuple(filter(truth, image_shape))
        images, labels = zip(*chunked_data)

        with h5py.File(data_path, 'a') as hdf:
            new_X_len, new_y_len = map(len, [images, labels])
            [group_label], group_labels = spy(labels)
            person_group_name = str(group_label)

            if person_group_name not in hdf:
                g = hdf.create_group(person_group_name)
                X = g.create_dataset(images_ds_name, (0, *image_shape), chunks=True, maxshape=(None, *image_shape))
                y = g.create_dataset(labels_ds_name, (0,), chunks=True, maxshape=(None,))
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
        save_data_path: UNIVERSAL_PATH_TYPE,
        q: asyncio.Queue,
        image_frame_shape: FRAME_SHAPE_TYPE,
        metadata: UNIVERSAL_CONFIG_TYPE,
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
        camera_frame_shape: FRAME_SHAPE_TYPE,
        curr_label: int,
        person: UNIVERSAL_CONFIG_TYPE,
        save_data_path: str,
        max_buffer_len: int = 1E2,
        max_queue_size: int = 1E3,
        *,
        as_grayscale: bool = False

    ) -> None:

        """ (NEED TO VERIFY!) Processing of each frame - search for faces and save them into HDF5 storage with class label and meta-information

            :param camera_id: Camera system id
            :param camera_close_key: Keyboard button key for exit current camera session
            :param haar_face_cascade: Haar cascade model object
            :param camera_frame_shape: Camera frame shape (width, height and color channel)
            :param curr_label: Current class label to identify a person's face
            :param person: Person`s meta-information
            :param save_data_path: Path to HDF5 storage
            :param max_buffer_len: Maximum number of extracted faces in the RAM buffer before storing in storage
            :param max_queue_size: Maximum size of async saving task Queue (NEED TO VERIFY!)
        """

        cam = cv2.VideoCapture(camera_id)

        with ProcessPoolExecutor(1) as pool:
            tasks, chunked_data = deque(), deque()
            q = asyncio.Queue(maxsize=max_queue_size)
            try:
                shift = 0
                is_live = True
                while is_live:
                    is_live = not (cv2.waitKey(30) == ord(camera_close_key))  ### control of exit button pressing
                    _, frame = cam.read()
                    faces_coords = ImageWorking.get_faces_area(haar_face_cascade, frame)

                    if faces_coords:
                        first_face_coords = itemgetter(0, 1, 2, 3)(faces_coords[0])  ### Use first face only
                        first_face_coords = RectArea(*first_face_coords)
                        ImageWorking.draw_rect_area(frame, first_face_coords)
                        frame_area = ImageWorking.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape, as_grayscale=as_grayscale)

                        ### If image buffer exceeds the limit or exit button is pressed - save the images asynchronously
                        if (shift > max_buffer_len) or not is_live:
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

                    cv2.imshow('frame', frame)

            finally:
                ### Wait for all assigned async tasks to complete and then release all resources
                await asyncio.gather(*tasks, return_exceptions=True)
                cv2.destroyAllWindows()
                cam.release()


    @staticmethod
    def create_datasets(
        persons: UNIVERSAL_CONFIG_TYPE,
        camera: UNIVERSAL_CONFIG_TYPE,
        haar_cascade_path: UNIVERSAL_PATH_TYPE,
        save_data_path: UNIVERSAL_PATH_TYPE,
        as_grayscale: bool = False

    ) -> None:

        """ Create classify frames frames from camera

            :param persons: Mapping of persons labels and meta-information
            :param camera: Camera config
            :param haar_cascade_path: Path to Haar-cascade configuration file
            :param save_data_path: HDF5 storage to save extracted faces and meta-information
            :param as_grayscale: Extract person faces as grayscale
        """

        camera_id, camera_frame_shape, camera_close_key = itemgetter('camera_id', 'camera_frame_shape', 'camera_close_key')(camera)
        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)

        for label, person in sorted(persons.items(), key=itemgetter(0)):
            inp_msg = "{0} {1}, are you ready? [Y/n]: ".format(*map(str.capitalize, always_iterable(itemgetter('first_name', 'last_name')(person))))
            inp_val = input(inp_msg)
            if inp_val.lower() == 'y':
                asyncio.run(TrainsetCreation._create_stage_dataset(camera_id, camera_close_key, haar_face_cascade, camera_frame_shape, int(label), person, save_data_path, as_grayscale=as_grayscale))




class LearnModel:
    """ Model compile and fit utils """

    @staticmethod
    def __prepare_model(d: Mapping, input_shape: FRAME_SHAPE_TYPE, n_classes: int) -> MODEL_CONFIG_TYPE:
        """ Prepare model if needed (batch shape and classes count parameters)

            :param d: Model config from json
            :param input_shape: Input train data shape for implementation into model
            :param n_classes: Classes count for implementation into model
        """
        dd = defaultdict(lambda: defaultdict(dict), **d)  # type: dict
        dd['config']['layers'][0]['config']['batch_input_shape'] = [None, *input_shape]
        dd['config']['layers'][-1]['config']['units'] = n_classes
        return dd


    @staticmethod
    def _create_model(config_path: UNIVERSAL_PATH_TYPE, input_shape: FRAME_SHAPE_TYPE, n_classes: int) -> Model:
        """ Create, prepare and compile KEras classify model

            :param config_path: Path to model json config
            :param input_shape: Input train data shape
            :param n_classes: Classes count
        """
        with open(config_path,  'r') as f:
            model_config = LearnModel.__prepare_model(json.load(f), input_shape, n_classes)
            model = deserialize(model_config)
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            return model


    @staticmethod
    def fit_model(
        model: Model,
        train_data: TRAIN_DATA_GEN_TYPE,
        X_test: TRAIN_DATA_TYPE = None,
        y_test: TRAIN_LABELS_TYPE = None,
        epochs: int = 1,
        steps_per_epoch: Union[int, None] = None,
        shuffle: bool = True,
        verbose_mode: bool = True

    ) -> Model:

        """ Train and return fitted model

            :param model: Configured Keras model object
            :param train_data: Train data object (Keras <Sequence> or iterable with train data and labels)
            :param X_test: Test data
            :param y_test: Test labels
            :param epochs: Count of epoch to learn
            :param steps_per_epoch: Count of step in one train epoch
            :param shuffle: Is shuffle train data ?
            :param verbose_mode: Verbose learn output ?
        """

        if isinstance(train_data, (Generator, KerasSequence)):
            model.fit_generator(generator=train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, shuffle=shuffle, verbose=verbose_mode)
        else:
            X, y = ( np.array(tuple(collapse(x, levels=1))) for x in zip(*train_data) )
            model.fit(X, y, epochs=epochs, shuffle=shuffle, verbose=verbose_mode)

        if X_test and y_test:
            model.evaluate(X_test, y_test)

        return model


    @staticmethod
    def start_learn(
        train_data: TRAIN_DATA_GEN_TYPE,
        model_config_path: UNIVERSAL_PATH_TYPE,
        camera_config: UNIVERSAL_CONFIG_TYPE,
        learn_config: UNIVERSAL_CONFIG_TYPE,
        n_classes: int,
        verbose_mode: bool = True

    ) -> Model:

        """ Configure and fit model from training dataset

            :param train_data: Train data object (Keras <Sequence> or iterable with train data and labels)
            :param camera_config: Camera config
            :param learn_config: Learn process config
            :param model_config_path: Path to json model config
            :param n_classes: Count of different classes to learn model
            :param verbose_mode: Verbose learn output ?
        """

        frame_shape = itemgetter('camera_frame_shape')(camera_config)
        epochs, steps_per_epoch, shuffle = itemgetter('epochs', 'steps_per_epoch', 'shuffle')(learn_config)

        model = LearnModel._create_model(model_config_path, frame_shape, n_classes)
        return LearnModel.fit_model(model, train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, shuffle=shuffle, verbose_mode=verbose_mode)




class PredictModel:
    """ Model learning and recognition processing """

    @staticmethod
    def _face_predict(model, label_mapping: Mapping, frame: np.ndarray, extracted_face: np.ndarray, face_coords: RectArea) -> np.ndarray:
        """ Predict extracted face and draw label with this name

            :param model: Trained Keras model
            :param label_mapping: Mapping persons labels
            :param frame: Current image frame
            :param extracted_face: Part of frame with extracted face
            :param face_coords: Coords of extracted face on current frame
        """

        ### Predict label ###
        predicted_code = model.predict_classes( np.array([extracted_face]) )
        predicted_meta = label_mapping.get(str(first(predicted_code, None)), {})

        ### Draw face area ###
        ImageWorking.draw_rect_area(frame, face_coords)

        ### Draw person info ###
        for i, (k, v) in enumerate(predicted_meta.items()):
            msg = "{0}: {1}".format(k, v)
            text = cv2.putText(frame, msg, (face_coords.x2, face_coords.y2-i*20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', text)

        return predicted_code


    @staticmethod
    async def start_predict(
        persons_mapping: UNIVERSAL_CONFIG_TYPE,
        camera: UNIVERSAL_CONFIG_TYPE,
        model_path: UNIVERSAL_PATH_TYPE,
        haar_cascade_path: UNIVERSAL_PATH_TYPE,
        *, as_grayscale: bool = False

    ) -> None:

        """ Extract and predict faces from each camera frame

            :param persons_mapping: Mapping persons labels and metadata
            :param camera: Camera config
            :param model_path: Path to trained Keras model
            :param haar_cascade_path: Path to Haar cascade config file
            :param as_grayscale: Predict faces as grayscale
        """

        camera_id = camera['camera_id']
        camera_frame_shape = tuple(camera['camera_frame_shape'])
        camera_close_key = camera['camera_close_key']

        haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        model = load_model(model_path)
        predict_func = partial(PredictModel._face_predict, model, persons_mapping)
        cam = cv2.VideoCapture(camera_id)
        # cam.set(3, 800)
        # cam.set(4, 600)

        while True:
            _, frame = cam.read()
            cv2.imshow('frame', frame)

            faces_coords = ImageWorking.get_faces_area(haar_face_cascade, frame)
            if faces_coords:
                first_face_coords = RectArea(*itemgetter(0, 1, 2, 3)(faces_coords[0]))  ### Use first face only
                extracted_face = ImageWorking.extract_frame_area(frame, first_face_coords, frame_reshape=camera_frame_shape, as_grayscale=as_grayscale)
                predict_func(frame, extracted_face, first_face_coords)

            if cv2.waitKey(30) == ord(camera_close_key):
                break

        cam.release()
        cv2.destroyAllWindows()
