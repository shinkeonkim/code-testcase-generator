# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import proto.executor_pb2 as executor__pb2

GRPC_GENERATED_VERSION = '1.71.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in executor_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class CodeExecutorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ExecuteCode = channel.unary_unary(
                '/code_executor.CodeExecutor/ExecuteCode',
                request_serializer=executor__pb2.ExecuteRequest.SerializeToString,
                response_deserializer=executor__pb2.ExecuteResponse.FromString,
                _registered_method=True)
        self.GetStatus = channel.unary_unary(
                '/code_executor.CodeExecutor/GetStatus',
                request_serializer=executor__pb2.StatusRequest.SerializeToString,
                response_deserializer=executor__pb2.StatusResponse.FromString,
                _registered_method=True)


class CodeExecutorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ExecuteCode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CodeExecutorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ExecuteCode': grpc.unary_unary_rpc_method_handler(
                    servicer.ExecuteCode,
                    request_deserializer=executor__pb2.ExecuteRequest.FromString,
                    response_serializer=executor__pb2.ExecuteResponse.SerializeToString,
            ),
            'GetStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.GetStatus,
                    request_deserializer=executor__pb2.StatusRequest.FromString,
                    response_serializer=executor__pb2.StatusResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'code_executor.CodeExecutor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('code_executor.CodeExecutor', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class CodeExecutor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ExecuteCode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/code_executor.CodeExecutor/ExecuteCode',
            executor__pb2.ExecuteRequest.SerializeToString,
            executor__pb2.ExecuteResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/code_executor.CodeExecutor/GetStatus',
            executor__pb2.StatusRequest.SerializeToString,
            executor__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
