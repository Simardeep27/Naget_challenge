from atomic_agents import BaseTool, BaseToolConfig, AtomicAgent, AgentConfig
from schemas import (
    IntentOutSchema,
    UserRequest,
)
from pydantic import Field
from litellm import acompletion
from dotenv import load_dotenv
import os
import json



class PlanRequestToolConfig(BaseToolConfig):
    llm_key: str = Field(..., description="LiteLLM proxy API key ")
    model_name : str = Field(..., description = "MODEL NAME")


class PlanRequestTool(BaseTool[UserRequest, IntentOutSchema]):
    input_schema = UserRequest
    output_schema = IntentOutSchema

    def __init__(self, config: PlanRequestToolConfig):
        super().__init__(config)
        self.model_name = config.model_name


    async def run(self, params: UserRequest, parallel: bool = False) -> IntentOutSchema:
        schema_def = json.dumps(IntentOutSchema.model_json_schema(), indent=2)
        system = (
            "You are an intent planner for a web-information gatherer.\n"
            "Goal: convert the user query into a short list of search-ready intent requests.\n"
            "Rules:\n"
            "- Each intent must be a standalone natural-language request.\n"
            "- Make intents non-overlapping (no near-duplicates).\n"
            "- Prefer action verbs: Find / Compare / List / Explain / Summarize / Extract.\n"
            "- No numbering, no extra keys, no commentary. Output must match the schema: {schema_def}.\n"
        )

        user = (
            "Create intent_requests for the following query.\n\n"
            f"QUERY:\n{params['user_request']}\n"
        )

        resp = await acompletion(
            model=f"deepinfra/{self.model_name}",
            messages=[
                {"role": "system", "content": system.format(schema_def= schema_def)},
                {"role": "user", "content": user},
            ],
            response_format={ "type": "json_object" },
            temperature=0.2
        )
        content = (resp.choices[0].message.content or "")
        out = IntentOutSchema.model_validate_json(content)
        return out