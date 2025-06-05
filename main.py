import json
import os
import logging
import random
import grpc
import requests
from proto import executor_pb2
from proto import executor_pb2_grpc
from typing import Dict, Any, List
from dotenv import load_dotenv
import re
from pydantic import BaseModel
import openai

# .env 파일 자동 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CODE_EXECUTOR_ADDR = os.getenv("CODE_EXECUTOR_ADDR", "localhost:50051")

# Gemini API 설정
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class InputExamples(BaseModel):
    examples: List[str]

# 샘플 문제 데이터 로딩 (테스트용)
def load_sample_problem():
    with open("examples/sample_problem.json", "r") as f:
        return json.load(f)

# LLM(Gemini)로 input_range 추출 함수
def extract_input_range_with_llm(description: str, input_description: str) -> Dict[str, Dict[str, int]]:
    prompt = f"""
아래는 프로그래밍 문제의 입력 설명입니다.
각 입력 변수의 이름과 그 값의 범위를 JSON 형태로 추출해 주세요.
입력 설명(HTML):\n{input_description}
문제 설명(HTML):\n{description}
출력 예시:\n{{\n  \"A\": {{\"min\": 1, \"max\": 9}},\n  \"B\": {{\"min\": 1, \"max\": 9}}\n}}
"""
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            import re
            import ast
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return ast.literal_eval(match.group(0))
                except Exception as e:
                    logger.warning(f"Gemini input_range 파싱 실패: {e}, text={text}")
                    raise RuntimeError(f"Gemini input_range 파싱 실패: {e}, text={text}")
            logger.warning(f"Gemini input_range 추출 실패, text={text}")
            raise RuntimeError(f"Gemini input_range 추출 실패, text={text}")
        else:
            logger.error(f"Gemini API(input_range) 호출 실패: {response.status_code}, {response.text}")
            raise RuntimeError(f"Gemini API(input_range) 호출 실패: {response.status_code}, {response.text}")
    except Exception as e:
        logger.error(f"Gemini API(input_range) 호출 중 예외: {e}")
        raise RuntimeError(f"input_range LLM 생성 실패: {e}")

# C++ 코드 실행 함수 (validation/solution)
def execute_cpp_code(code: str, input_str: str, timeout: int = 5) -> Dict[str, Any]:
    try:
        with grpc.insecure_channel(CODE_EXECUTOR_ADDR) as channel:
            stub = executor_pb2_grpc.CodeExecutorStub(channel)
            req = executor_pb2.ExecuteRequest(
                code=code,
                language="cpp",
                version="23",
                timeout_seconds=timeout,
                memory_limit_mb=128,
                input=[input_str]
            )
            resp = stub.ExecuteCode(req)
            logger.info(f"C++ 실행 결과: status={resp.status}, stdout={resp.stdout.strip()}, stderr={resp.stderr.strip()}, error={resp.error_message.strip()}")
            return {
                "status": resp.status,
                "stdout": resp.stdout.strip(),
                "stderr": resp.stderr.strip(),
                "error_message": resp.error_message.strip()
            }
    except Exception as e:
        logger.error(f"C++ 코드 실행 중 예외: {e}")
        return {"status": -1, "stdout": "", "stderr": str(e), "error_message": str(e)}

# LLM(Gemini)로 validation_code 추출 함수
def extract_validation_code_with_llm(description: str, input_description: str) -> str:
    prompt = f"""
아래는 프로그래밍 문제의 입력 설명과 문제 설명입니다.
입력값의 유효성을 검증하는 C++ 23 기준의 main 함수 전체 코드를 생성해 주세요.
반드시 입력은 표준 입력(예: std::cin)으로 받고, assert로 검증하세요.
입력 설명(HTML):\n{input_description}
문제 설명(HTML):\n{description}
출력 예시 (두 수를 입력받고 더한 값을 출력하는 문제에 대해 유효성을 검증하는 코드는 아래와 같이 작성됩니다.):
#include <cassert>\n#include <iostream>\nint main() {{  int A, B;\n  std::cin >> A >> B;\n  assert(1 <= A && A <= 9);\n  assert(1 <= B && B <= 9);\n  return 0;\n}}
"""
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            import re
            match = re.search(r'```(?:cpp|c\+\+)?([\s\S]*?)```', text)
            if match:
                return match.group(1).strip()
            return text.strip()
        else:
            logger.error(f"Gemini API(validation_code) 호출 실패: {response.status_code}, {response.text}")
            raise RuntimeError(f"Gemini API(validation_code) 호출 실패: {response.status_code}, {response.text}")
    except Exception as e:
        logger.error(f"Gemini API(validation_code) 호출 중 예외: {e}")
        raise RuntimeError(f"validation_code LLM 생성 실패: {e}")

# LLM(Gemini)로 solution_code 추출 함수
def extract_solution_code_with_llm(description: str, input_description: str, output_description: str) -> str:
    prompt = f"""
아래는 프로그래밍 문제의 설명, 입력 설명, 출력 설명입니다.
C++ 23 기준의 정답 코드를 main 함수 전체로만 생성해 주세요.
반드시 입력은 표준 입력(예: std::cin)으로 받고, 출력을 표준 출력(예: std::cout)으로 하세요.
문제 설명(HTML):\n{description}
입력 설명(HTML):\n{input_description}
출력 설명(HTML):\n{output_description}
출력 예시(두 수를 입력받고 더한 값을 출력하는 문제에 대해 정답 코드는 아래와 같이 작성됩니다.):
#include <iostream>\nint main() {{\n  int A, B;\n  std::cin >> A >> B;\n  std::cout << (A+B) << std::endl;\n  return 0;\n}}
"""
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        if response.status_code == 200:
            result = response.json()
            text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            import re
            match = re.search(r'```(?:cpp|c\+\+)?([\s\S]*?)```', text)
            if match:
                return match.group(1).strip()
            return text.strip()
        else:
            logger.error(f"Gemini API(solution_code) 호출 실패: {response.status_code}, {response.text}")
            raise RuntimeError(f"Gemini API(solution_code) 호출 실패: {response.status_code}, {response.text}")
    except Exception as e:
        logger.error(f"Gemini API(solution_code) 호출 중 예외: {e}")
        raise RuntimeError(f"solution_code LLM 생성 실패: {e}")

def generate_inputs_with_llm(description: str, input_description: str, num_testcases: int, example_input_datas: list[str]) -> list[str]:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"아래는 프로그래밍 문제의 입력 설명과 문제 설명입니다.\n"
        f"이 문제에 대해 유효한 입력 예시를 {num_testcases}개, 각 예시는 여러 줄일 수 있습니다.\n"
        f"각 입력 예시마다 하나의 문자열로 담기게 해야합니다. 따라서 여러 개의 입력 예시를 문자열 목록(list of string)으로 반환해 주세요.\n"
        f"입력 설명(HTML):\n{input_description}\n"
        f"문제 설명(HTML):\n{description}\n"
        f"입력 예시 목록:\n{example_input_datas}\n"
    )
    function_schema = {
        "name": "InputExamples",
        "description": "문제 입력 예시 목록",
        "parameters": InputExamples.model_json_schema(),
    }
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": function_schema}],
        tool_choice="auto",
    )
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    return arguments["examples"][:num_testcases]

def get_attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)

# 테스트케이스 생성 함수 (validation, solution_code 실행)
def generate_testcases(problem: Any) -> Dict:
    description = get_attr(problem, "description")
    input_description = get_attr(problem, "input_description")
    output_description = get_attr(problem, "output_description")
    validation_code = get_attr(problem, "validation_code")
    solution_code = get_attr(problem, "solution_code")
    num_testcases = get_attr(problem, "num_testcases")
    example_testcases = get_attr(problem, "example_testcases")

    if not validation_code:
        validation_code = extract_validation_code_with_llm(description, input_description)
        logger.info(f"LLM으로 생성된 validation_code: {validation_code}")
    if not solution_code:
        solution_code = extract_solution_code_with_llm(description, input_description, output_description)
        logger.info(f"LLM으로 생성된 solution_code: {solution_code}")

    if example_testcases and len(example_testcases) > 0:
        if isinstance(example_testcases[0], dict):
            example_inputs = [x["input"] for x in example_testcases]
        else:
            example_inputs = [x.input for x in example_testcases]
    else:
        example_inputs = []

    input_examples = generate_inputs_with_llm(
        description,
        input_description,
        num_testcases,
        example_inputs
    )
    print("생성된 전체 입력 예시 수: ", len(input_examples))
    print("생성된 입력 예시: ", input_examples)
    testcases = []
    for input_str in input_examples:
        val_result = execute_cpp_code(validation_code, input_str)
        if val_result["status"] != 2 or val_result["stderr"] or val_result["error_message"]:
            logger.info(f"[검증 실패] input='{input_str}', result={val_result}")
            continue
        sol_result = execute_cpp_code(solution_code, input_str)
        if sol_result["status"] != 2 or sol_result["stderr"] or sol_result["error_message"]:
            logger.info(f"[정답코드 실패] input='{input_str}', result={sol_result}")
            continue
        testcase = {
            "input": input_str,
            "output": sol_result["stdout"]
        }
        testcases.append(testcase)
    logger.info(f"생성된 테스트케이스 수: {len(testcases)}")

    return {
        "validation_code": validation_code,
        "solution_code": solution_code,
        "testcases": testcases,
    }
