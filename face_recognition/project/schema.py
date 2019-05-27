# coding: utf-8



SCHEMA_MAPPING = {
    "persons": {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                    },
                    "patternProperties": {
                        ".+": {"type": ["integer", "string"]}
                    },
                    "required": ["first_name", "last_name"]
                }
            }
        },
        "required": ["data"]
    },

    "camera": {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "camera_id": {"type": "integer"},
                    "camera_close_key": {"type": "string"},
                    "camera_frame_shape": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["camera_id", "camera_close_key", "camera_frame_shape"]
            }
        },
        "required": ["data"]
    },

    "model_config": {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {
            "class_name": {"type": "string"},
            "config": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "layers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class_name": {"type": "string"},
                                "config": {
                                    "type": "object"
                                }
                            }
                        }
                    }
                }
            },
            "keras_version": {"type": "string"},
            "backend": {"type": "string", "enum": ["theano", "tensorflow"]}
        }
    }
}
