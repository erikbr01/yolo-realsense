# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/detection_msg.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aprotos/detection_msg.proto\x12\x06Vision\"y\n\tDetection\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x0c\n\x04roll\x18\x04 \x01(\x02\x12\r\n\x05pitch\x18\x05 \x01(\x02\x12\x0b\n\x03yaw\x18\x06 \x01(\x02\x12\r\n\x05label\x18\x07 \x01(\t\x12\x12\n\nconfidence\x18\x08 \x01(\x02')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, 'protos.detection_msg_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _DETECTION._serialized_start = 38
    _DETECTION._serialized_end = 159
# @@protoc_insertion_point(module_scope)
