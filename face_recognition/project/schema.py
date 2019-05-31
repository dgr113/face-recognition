# coding: utf-8



SCHEMA_MAPPING = {
    "persons": {
        "type": "object",
        "patternProperties": {
            r"\d+": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                },
                "patternProperties": {
                    r".+": {"type": ["integer", "string"]}
                },
                "required": ["first_name", "last_name"]
            }
        }
    },

    "camera": {
        "type": "object",
        "properties": {
            "camera_id": {"type": "integer"},
            "camera_close_key": {"type": "string"},
            "camera_frame_shape": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
        },
        "required": ["camera_id", "camera_close_key", "camera_frame_shape"]
    },

    "model_config": {
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
