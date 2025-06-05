# Code TestCase Generator 설계

## 1. 개요
- 다양한 프로그래밍 문제에 대해 자동으로 테스트케이스를 생성하고, 입력값 검증 및 정답 출력을 산출하는 시스템.
- gRPC 기반 통신, Rust 코드 실행, LLM API 연동, Docker 환경 지원.

## 2. 주요 컴포넌트

### 2.1 gRPC 서버 (Python)
- gRPC 서버로 동작하며, 외부에서 테스트케이스 생성 요청을 받음
- 입력: boj_id, description(HTML), input_description(HTML), output_description(HTML), validation_code(Rust, optional), solution_code(Rust, optional), 생성할 테스트케이스 개수
- 출력: 테스트케이스 리스트 (각각 boj_id, Input, Output 포함)

### 2.2 LLM 연동 모듈
- validation_code, solution_code, 테스트케이스 입력값이 없는 경우 LLM(Gemini/ChatGPT) API를 통해 자동 생성
- LLM 프롬프트 설계 및 결과 파싱

### 2.3 테스트케이스 생성 로직
- 문제 입력값 범위에 맞는 랜덤 입력값 생성 (input_range는 LLM으로 추출)
- 입력값 검증(validation_code 실행, code-executor 활용)
- 정답 코드(solution_code 실행, code-executor 활용)로 출력값 산출
- 실패(검증 오류)시 해당 테스트케이스는 결과에서 제외

### 2.4 code-executor 연동
- Rust 코드(validation_code, solution_code) 실행을 위해 code-executor와 연동
- Docker 환경에서 실행 지원

### 2.5 예제 및 테스트
- examples/ 폴더에 샘플 문제 및 테스트케이스 생성 예제 포함

## 3. 데이터 흐름
1. 클라이언트가 gRPC로 테스트케이스 생성 요청
2. 입력값/코드 미제공시 LLM을 통해 자동 생성
3. 랜덤 입력값 생성 및 LLM을 통한 테스트케이스 후보 생성
4. 각 입력값에 대해 validation_code 실행(실패시 제외)
5. solution_code 실행하여 출력값 산출
6. boj_id, Input, Output을 JSON 리스트로 반환

## 4. 확장성 및 관리
- Poetry로 의존성 관리
- Docker/Docker-compose로 배포 및 실행 환경 통일
- LLM API, code-executor 등 외부 서비스와의 연동 모듈화

## 5. 참고 파일 및 경로
- 문제 모델: AD_Project_athena/source_code/webapp/problem/models/problem.py
- LLM 연동: AD_Project_athena/source_code/webapp/agent/tasks/llm_feedback_task.py
- 테스트케이스 생성 참고: AD_Project_athena/source_code/webapp/article/services/article_collect_service.py
- code-executor 예제: code-executor/examples/test_client.rs 