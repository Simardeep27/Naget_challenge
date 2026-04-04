from __future__ import annotations

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from pydantic import Field

from schema import ResearchPlan, ResearchSlot
from utils.config import get_intent_model_name
from utils.llm_utils import run_structured_generation
from utils.text_utils import compact_text, normalize_key

LOCAL_QUERY_KEYWORDS = {
    "restaurant",
    "pizza",
    "cafe",
    "coffee",
    "bar",
    "hotel",
    "museum",
    "park",
    "attraction",
    "places",
    "things to do",
}
LIVE_EVENT_KEYWORDS = {
    "live",
    "currently",
    "current",
    "ongoing",
    "today",
    "now",
    "match",
    "matches",
    "game",
    "games",
    "fixture",
    "fixtures",
    "score",
    "scores",
    "football",
    "soccer",
    "nba",
    "nfl",
    "mlb",
    "nhl",
}
ORGANIZATION_QUERY_KEYWORDS = {
    "startup",
    "startups",
    "company",
    "companies",
    "organization",
    "organizations",
    "vendor",
    "vendors",
}
PRODUCT_QUERY_KEYWORDS = {
    "tool",
    "tools",
    "software",
    "database",
    "model",
    "models",
    "platform",
    "service",
    "services",
    "product",
    "products",
}
COMPARISON_QUERY_KEYWORDS = {
    "best",
    "top",
    "compare",
    "comparison",
    "vs",
    "versus",
    "rank",
    "ranking",
}


class ResearchPlannerInput(BaseIOSchema):
    """Input payload for building a slot-driven research plan."""

    user_request: str = Field(..., description="The user's research query")
    deep_research: bool = Field(
        default=False,
        description="Whether the user requested a deeper research pass",
    )
    lightning: bool = Field(
        default=False,
        description="Whether the user requested the lightning mode",
    )


class ResearchPlannerToolConfig(BaseToolConfig):
    """Model configuration for the research planner."""

    model_name: str = Field(default_factory=get_intent_model_name)


def _slot(
    key: str,
    label: str,
    description: str,
    *hints: str,
    required: bool,
) -> ResearchSlot:
    return ResearchSlot(
        key=normalize_key(key or label) or "slot",
        label=label,
        description=description,
        required=required,
        search_hints=[hint for hint in hints if compact_text(hint)],
    )


def _has_any_term(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _pluralize_entity_type(entity_type: str) -> str:
    if entity_type.endswith("match"):
        return f"{entity_type}es"
    if entity_type.endswith("y") and not entity_type.endswith(("ay", "ey", "iy", "oy", "uy")):
        return f"{entity_type[:-1]}ies"
    if entity_type.endswith("s"):
        return entity_type
    return f"{entity_type}s"


def build_fallback_research_plan(
    user_request: str,
    *,
    deep_research: bool,
    lightning: bool,
) -> ResearchPlan:
    lowered = user_request.lower()
    mode = "lightning" if lightning else ("deep" if deep_research else "standard")
    is_local_query = _has_any_term(lowered, LOCAL_QUERY_KEYWORDS) or " near " in lowered
    is_live_event_query = _has_any_term(lowered, LIVE_EVENT_KEYWORDS)
    is_organization_query = _has_any_term(lowered, ORGANIZATION_QUERY_KEYWORDS)
    is_product_query = _has_any_term(lowered, PRODUCT_QUERY_KEYWORDS)
    is_comparison_query = _has_any_term(lowered, COMPARISON_QUERY_KEYWORDS)

    required_slots: list[ResearchSlot] = [
        _slot(
            "name",
            "Name",
            "Name of the candidate entity being compared",
            "official name",
            "entity name",
            required=True,
        ),
        _slot(
            "summary",
            "Summary",
            "Short source-backed description of what the entity is",
            "overview",
            "what it is",
            required=True,
        ),
    ]
    nice_to_have_slots: list[ResearchSlot] = []
    constraints: list[str] = []
    evidence_types = [
        "official source",
        "independent review or ranking",
        "trusted directory or profile",
    ]
    entity_type = "entity"

    if is_live_event_query:
        entity_type = "match" if any(term in lowered for term in ("match", "matches", "fixture", "fixtures")) else "event"
        evidence_types = [
            "official schedule or organizer page",
            "live status or scoreboard",
            "news or live tracker",
        ]
        required_slots.extend(
            [
                _slot(
                    "participants",
                    "Participants",
                    "Teams, people, or sides involved in the event",
                    "teams",
                    "opponents",
                    required=True,
                ),
                _slot(
                    "status",
                    "Status",
                    "Current or scheduled status of the event",
                    "live",
                    "in progress",
                    "final",
                    required=True,
                ),
                _slot(
                    "start_time",
                    "Start Time",
                    "Scheduled or actual event time",
                    "kickoff",
                    "start time",
                    required=True,
                ),
                _slot(
                    "competition",
                    "Competition",
                    "League, tournament, or event grouping",
                    "league",
                    "competition",
                    required=True,
                ),
            ]
        )
        nice_to_have_slots.extend(
            [
                _slot(
                    "venue",
                    "Venue",
                    "Where the event is happening",
                    "venue",
                    "stadium",
                    required=False,
                ),
                _slot(
                    "score",
                    "Score",
                    "Current or final score if available",
                    "score",
                    required=False,
                ),
            ]
        )
    elif is_local_query:
        if any(term in lowered for term in ("restaurant", "pizza", "cafe", "coffee", "bar")):
            entity_type = "restaurant"
        elif any(term in lowered for term in ("hotel",)):
            entity_type = "hotel"
        else:
            entity_type = "place"
        evidence_types = [
            "official website or menu",
            "local review or ranking",
            "map or directory profile",
        ]
        required_slots.extend(
            [
                _slot(
                    "location",
                    "Location",
                    "Where the place is located",
                    "address",
                    "neighborhood",
                    "city",
                    required=True,
                ),
                _slot(
                    "category",
                    "Category",
                    "Type of place or cuisine",
                    "type",
                    "cuisine",
                    required=True,
                ),
                _slot(
                    "notable_signal",
                    "Notable Signal",
                    "Why this place stands out for the query",
                    "popular",
                    "best known for",
                    "review",
                    required=True,
                ),
            ]
        )
        nice_to_have_slots.extend(
            [
                _slot(
                    "website",
                    "Website",
                    "Official URL or menu page",
                    "official site",
                    "menu",
                    required=False,
                ),
                _slot(
                    "hours",
                    "Hours",
                    "Opening hours if clearly available",
                    "hours",
                    "open now",
                    required=False,
                ),
                _slot(
                    "price_range",
                    "Price Range",
                    "Typical price range if clearly available",
                    "price",
                    "cost",
                    required=False,
                ),
            ]
        )
    elif is_organization_query:
        if "startup" in lowered or "startups" in lowered:
            entity_type = "startup"
        elif "company" in lowered or "companies" in lowered:
            entity_type = "company"
        else:
            entity_type = "organization"
        evidence_types = [
            "official website",
            "company profile or directory",
            "news or profile coverage",
        ]
        required_slots.extend(
            [
                _slot(
                    "focus_area",
                    "Focus Area",
                    "Primary sector, product area, or specialization",
                    "sector",
                    "focus",
                    "specialty",
                    required=True,
                ),
                _slot(
                    "location",
                    "Location",
                    "Headquarters or primary location",
                    "headquarters",
                    "location",
                    required=True,
                ),
            ]
        )
        nice_to_have_slots.extend(
            [
                _slot(
                    "website",
                    "Website",
                    "Official company site",
                    "official site",
                    required=False,
                ),
                _slot(
                    "funding_signal",
                    "Funding Signal",
                    "Funding, scale, or traction if clearly available",
                    "funding",
                    "backed",
                    "raised",
                    required=False,
                ),
            ]
        )
    elif is_product_query:
        if "database" in lowered:
            entity_type = "database"
        elif "model" in lowered:
            entity_type = "model"
        elif "platform" in lowered:
            entity_type = "platform"
        elif "service" in lowered or "services" in lowered:
            entity_type = "service"
        else:
            entity_type = "tool"
        evidence_types = [
            "official website or documentation",
            "independent comparison or review",
            "directory or profile",
        ]
        required_slots.append(
            _slot(
                "category",
                "Category",
                "Type of product or tool",
                "category",
                "type",
                required=True,
            )
        )
        nice_to_have_slots.append(
            _slot(
                "website",
                "Website",
                "Official URL or documentation entry point",
                "official site",
                "docs",
                required=False,
            )
        )
    elif is_comparison_query:
        required_slots.append(
            _slot(
                "notable_signal",
                "Notable Signal",
                "Core comparison signal that makes the entity relevant",
                "best",
                "rating",
                "review",
                required=True,
            )
        )

    if "open source" in lowered:
        constraints.append("Open-source or source-available candidates only")
        evidence_types.append("repository or source page")
        required_slots.append(
            _slot(
                "license",
                "License",
                "License or open-source status",
                "license",
                "open source",
                required=True,
            )
        )

    if any(term in lowered for term in ("latency", "low latency", "fast")):
        required_slots.append(
            _slot(
                "latency",
                "Latency",
                "Latency or speed signal relevant to production use",
                "latency",
                "response time",
                "throughput",
                required=True,
            )
        )

    if any(
        term in lowered
        for term in (
            "benchmark",
            "quality",
            "accuracy",
            "performance",
            "rating",
            "review",
            "best",
            "top",
        )
    ):
        required_slots.append(
            _slot(
                "quality_signal",
                "Quality Signal",
                "Evidence showing why the entity performs well or is highly regarded",
                "benchmark",
                "evaluation",
                "quality",
                required=True,
            )
        )

    if any(term in lowered for term in ("production", "deploy", "deployment", "hosted", "self-hosted")):
        required_slots.append(
            _slot(
                "deployment_constraints",
                "Deployment Constraints",
                "Production deployment modes, limits, or infra requirements",
                "deployment",
                "self-hosted",
                "API",
                "inference",
                required=True,
            )
        )

    if any(term in lowered for term in ("pricing", "cost", "cheap")):
        required_slots.append(
            _slot(
                "pricing",
                "Pricing",
                "Pricing or cost signal if relevant",
                "pricing",
                "cost",
                required=True,
            )
        )

    if "github" in lowered or "readme" in lowered or "repository" in lowered:
        evidence_types.append("repository/readme")

    if not any(slot.key == "website" for slot in nice_to_have_slots):
        nice_to_have_slots.append(
            _slot(
                "website",
                "Website",
                "Official URL or primary source page",
                "official site",
                "website",
                required=False,
            )
        )
    if not any(slot.key == "category" for slot in nice_to_have_slots):
        nice_to_have_slots.append(
            _slot(
                "category",
                "Category",
                "High-level category or positioning",
                "type",
                "category",
                required=False,
            )
        )

    # Keep required slots compact and unique.
    deduped_required: list[ResearchSlot] = []
    seen_required: set[str] = set()
    for slot in required_slots:
        if slot.key in seen_required:
            continue
        deduped_required.append(slot)
        seen_required.add(slot.key)

    deduped_optional: list[ResearchSlot] = []
    for slot in nice_to_have_slots:
        if slot.key in seen_required or slot.key in {item.key for item in deduped_optional}:
            continue
        deduped_optional.append(slot)

    return ResearchPlan(
        user_query=user_request,
        research_depth=mode,
        objective=f"Compare {_pluralize_entity_type(entity_type)} relevant to: {user_request}",
        entity_type=entity_type,
        required_slots=deduped_required[:5],
        nice_to_have_slots=deduped_optional[:3],
        constraints=constraints,
        evidence_types=list(dict.fromkeys(evidence_types)),
    )


class ResearchPlannerTool(BaseTool[ResearchPlannerInput, ResearchPlan]):
    """Build a slot-driven research plan instead of many independent intent splits."""

    def __init__(
        self, config: ResearchPlannerToolConfig = ResearchPlannerToolConfig()
    ):
        super().__init__(config)
        self.model_name = config.model_name

    def run(self, params: ResearchPlannerInput) -> ResearchPlan:
        mode = "lightning" if params.lightning else ("deep" if params.deep_research else "standard")
        system_prompt = """
You turn a user query into a slot-driven research plan for a comparison agent.
Do not decompose the query into many independent intents unless they are truly disjoint.
Instead, represent the work as one objective with required slots, nice-to-have slots,
constraints, and evidence types.
Return valid JSON matching the provided schema.
Use 3 to 5 required slots and at most 3 nice-to-have slots.
Always include a required "name" slot and focus on fields that are directly useful for comparison.
Slot keys must be snake_case.
""".strip()
        user_prompt = f"""
User query:
{params.user_request}

Mode:
{mode}

Return a ResearchPlan with:
- a single comparison objective
- required_slots
- nice_to_have_slots
- constraints
- evidence_types
- entity_type specific to the user query

Avoid turning the plan into a list of separate search intents.
""".strip()

        try:
            return run_structured_generation(
                response_schema=ResearchPlan,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
            )
        except Exception:
            return build_fallback_research_plan(
                params.user_request,
                deep_research=params.deep_research,
                lightning=params.lightning,
            )
