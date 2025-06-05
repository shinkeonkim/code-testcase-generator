import grpc
from concurrent import futures
from proto import testcase_generator_pb2
from proto import testcase_generator_pb2_grpc
from main import generate_testcases
import json

class TestcaseGeneratorServicer(testcase_generator_pb2_grpc.TestcaseGeneratorServicer):
    def GenerateTestcases(self, request, context):
        # 요청을 dict로 변환
        problem = {
            "boj_id": request.boj_id,
            "description": request.description,
            "input_description": request.input_description,
            "output_description": request.output_description,
            "validation_code": request.validation_code,
            "solution_code": request.solution_code,
            "num_testcases": request.num_testcases
        }
        result = generate_testcases(problem)
        response = testcase_generator_pb2.GenerateTestcaseResponse()
        for tc in result['testcases']:
            response.testcases.add(
                input=tc["input"],
                output=tc["output"]
            )
        response.validation_code = result["validation_code"]
        response.solution_code = result["solution_code"]        
        response.boj_id = problem["boj_id"]

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    testcase_generator_pb2_grpc.add_TestcaseGeneratorServicer_to_server(TestcaseGeneratorServicer(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    print("gRPC TestcaseGenerator server started on port 50053")
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 
