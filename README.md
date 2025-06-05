# Code TestCase Generator

## PLAN

### Spec
- gRPC를 이용한 통신
- Python
- Poetry
- Docker, docker-compose
- LLM API (Gemini or ChatGPT)
- examples 폴더 내부에 테스트해볼 수 있는 파일 하나 만들어두기

### Input

- Problem ID
- 문제 본문, 입력, 출력 정보 등
  - /Users/koa/kmu-homework/AD_Project_athena/source_code/webapp/problem/models/problem.py 참고
- 문제 입력값 범위에 대한 assert만 수행하는 코드 (validation_code) , Optional, 입력으로 주어지지 않은 경우 LLM읉 통해 생성 (Rust Code)
- 정답 코드 (solution_code), Optional, 입력으로 주어지지 않은 경우 LLM읉 통해 생성
- 생성할 테스트케이스 개수

### 과정

- 주어지는 입력값에 따라, LLM을 통해 테스트케이스 개수만큼 테스트케이스를 생성합니다.
    - 이 과정에서 JSON 구조를 통해 가져오도록 합니다.
    - 참고: /Users/koa/kmu-homework/AD_Project_athena/source_code/webapp/article/services/article_collect_service.py
    - 테스트케이스 생성시 문제 범위에 맞고 최대한 랜덤 숫자를 활용하도록 랜덤 숫자를 먼저 생성하고 넘겨주는 등의 LLM 한계를 극복하도록 합니다.
- code-executor를 실행하여 validation_code를 실행하고 오류가 나는 경우 결과로 반환하지 않는다.
- code-executor를 실행하여 정확한 실행값을 가져온다.
  - 참고: /Users/koa/kmu-homework/code-executor/examples/test_client.rs

### Output

- 아래 항목을 json list 안에 각 요소마다 담아서 반환한다.
    - Problem ID
    - Input
    - Output

---

## 실행 가이드

### 1. 환경 변수
- `GEMINI_API_KEY`: GEMINI_API_KEY 키를 반드시 환경변수로 지정해야 합니다.

### 2. Docker Compose로 실행
```bash
# 환경변수 파일(.env) 생성
cat <<EOF > .env
GEMINI_API_KEY=xxx
EOF

docker-compose up --build
```

### 3. gRPC 서비스 엔드포인트
- gRPC: 0.0.0.0:50053 (proto: testcase_generator.proto)

### 4. gRPC 예제 요청 (Python)
```python
import grpc
import testcase_generator_pb2
import testcase_generator_pb2_grpc

channel = grpc.insecure_channel('localhost:50053')
stub = testcase_generator_pb2_grpc.TestcaseGeneratorStub(channel)
req = testcase_generator_pb2.GenerateTestcaseRequest(
    boj_id=1000,
    description="...",
    input_description="...",
    output_description="...",
    validation_code="...",
    solution_code="...",
    num_testcases=5
)
resp = stub.GenerateTestcases(req)
for tc in resp.testcases:
    print(tc.boj_id, tc.input, tc.output)
```
