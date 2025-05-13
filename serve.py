import os
from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="GPT2Deployment")
@serve.ingress(app)
class GPT2Deployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        lora_modules: Optional[List[str]] = None,
        request_logger: Optional[RequestLogger] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.lora_modules = lora_modules
        self.request_logger = request_logger

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_completion: Optional[OpenAIServingCompletion] = None

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        if not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            served_model_names = (
                self.engine_args.served_model_name or [self.engine_args.model]
            )
            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine,
                model_config,
                served_model_names,
                lora_modules=self.lora_modules,
                request_logger=self.request_logger,
            )

        logger.info(f"Request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_args(cli_args: Dict[str, str]):
    arg_parser = FlexibleArgumentParser(description="vLLM OpenAI API for GPT2")
    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for k, v in cli_args.items():
        arg_strings.extend([f"--{k}", str(v)])
    return parser.parse_args(arg_strings)


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    parsed_args = parse_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    return GPT2Deployment.bind(
        engine_args,
        parsed_args.lora_modules,
        cli_args.get("request_logger"),
    )


model = build_app(
    {
        "model": os.environ["MODEL_ID"],
        "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
        "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
    }
)
