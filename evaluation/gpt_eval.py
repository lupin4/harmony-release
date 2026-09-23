"""GPT-based reference-free evaluation, ported from VIGA evaluate_baseline.py.

Given a GENERATED image and a TARGET reference image (+ optional task
description), asks a GPT model to rate the generated image on a set of
criteria, prioritizing geometric structure / spatial layout / object
identity over stylistic differences.

The API key is read from ``OPENAI_API_KEY`` env var. To override the criteria
or prompt template, edit ``EVALUATION_CRITERIA`` / ``INSTRUCTION_TEMPLATE``.

Typical use:
    from gpt_eval import evaluate_pair
    result = evaluate_pair(
        pred_path="render.png",
        target_path="input.jpg",
        task_description="bedroom with bed, nightstand, lamp",
    )
    print(result["average_score"], result["object_identity"]["score"], ...)
"""
import base64
import json as _json
import os
from typing import Optional

from openai import OpenAI


# ── Criteria ───────────────────────────────────────────────────────────────────

EVALUATION_CRITERIA = {
    "visual_quality": {
        "criteria": (
            "Assess overall rendering quality: lighting, shadows, surface materials, "
            "and whether the scene looks physically plausible. "
            "Do NOT penalise for colour/texture/style differences that don't affect "
            "physical realism — focus on whether the lighting and geometry look correct."
        ),
        "scale": 5,
    },
    "object_identity": {
        "criteria": (
            "Check whether every object visible in the TARGET image also appears in "
            "the GENERATED image at the correct screen region, as two sub-checks: "
            ""
            "(A) FLOOR / FREESTANDING OBJECTS (sofas, chairs, tables, beds, lamps, "
            "plants, rugs, etc.): mentally divide both images into a 3x3 grid. For "
            "each floor object in the TARGET, note which cell(s) its bbox occupies, "
            "then verify the GENERATED image has a same-category object in those "
            "same cell(s). "
            "Bad: TARGET has a sofa in centre-bottom but GENERATED has a bed there; "
            "TARGET has a floor lamp on the right but GENERATED has nothing there. "
            ""
            "(B) WALL-MOUNTED OBJECTS (paintings, mirrors, windows, doors, shelves, "
            "wall sconces, clocks, radiators): for each wall-mounted object in the "
            "TARGET, check whether a same-category object appears on the same wall "
            "in the GENERATED image. Penalise if the object is entirely absent. "
            "Bad: TARGET has a painting on the back wall but GENERATED has a bare "
            "wall there; TARGET has a mirror but it is missing from GENERATED. "
            "Do NOT penalise: different frame style, different size, simplified "
            "representation — as long as something of the same category is present "
            "on the same wall. "
            ""
            "Judge both sub-checks by silhouette/bbox and screen region only — "
            "completely ignore colour, material, texture, and style differences. "
            "Before scoring, list every TARGET object (floor and wall-mounted) with "
            "its location and whether a match is present in GENERATED. "
            "Score 5 = every TARGET object (floor + wall) is present in the correct region. "
            "Score 3 = one object is missing, wrong type, or on the wrong wall/region. "
            "Score 1 = multiple objects are missing or clearly wrong."
        ),
        "scale": 5,
    },
    "object_orientation": {
        "criteria": (
            "Use the TARGET image as the ground truth for expected orientation. "
            "For each object, first determine from the TARGET which wall it is "
            "against (back wall / left wall / right wall) or whether it is a "
            "centre/floating item, then apply the matching rule below: "
            ""
            "(A) WALL-ADJACENT objects (sofa, bed, desk, cabinet, wardrobe, "
            "bookshelf against a wall): the front face must point AWAY from the "
            "wall it is backed against, i.e. facing into the room. "
            "Bad: sofa backed against the left wall but its front also faces the "
            "left wall — you can only see the sofa's back from the room. "
            "Good: sofa backed against the left wall and its front faces right, "
            "toward the room centre. "
            ""
            "(B) CENTRE / FLOATING items (dining table, island, coffee table with "
            "chairs around it): check whether each chair/stool faces toward the "
            "table it surrounds. "
            ""
            "(C) CAMERA-FACING orientation — use the TARGET to determine whether "
            "an object's front should face roughly toward the camera or away from it. "
            "If in the TARGET the object's front is visible from the camera viewpoint, "
            "the GENERATED image should also show the front. If in the TARGET you see "
            "the back of an object (e.g. a chair tucked under a desk facing away from "
            "camera), that is also correct in GENERATED. "
            "Bad: TARGET shows the front of an armchair but GENERATED shows only its "
            "back from the same viewpoint. "
            ""
            "Do NOT penalise ~45° angular offsets as long as the front half faces "
            "in the correct general direction. "
            "Before scoring, list each object, which wall it is against (or 'centre'), "
            "and whether its orientation in GENERATED matches the TARGET. "
            "Score 5 = all objects face the correct direction per the TARGET. "
            "Score 3 = one object is reversed relative to what the TARGET shows. "
            "Score 1 = most objects face the wrong direction."
        ),
        "scale": 5,
    },
    "spatial_accuracy": {
        "criteria": (
            "Evaluate three sub-aspects, then give one combined score: "
            "(A) Floor object region — are large floor objects (sofa, bed, desk, "
            "dining table) in the correct area of the room (back wall / left wall / "
            "right wall / centre) compared to the TARGET? "
            "(B) Wall-mounted objects — are paintings, mirrors, windows, doors, "
            "shelves, and sconces on the correct wall and at the correct height zone "
            "(lower third / mid / upper third of the wall)? Compare directly to TARGET. "
            "(C) Decoration placement — are small decorations (vases, lamps, books, "
            "plants, candles) placed ON or immediately beside the correct supporting "
            "surface (table top, shelf, floor next to sofa)? "
            "Bad: vase floating at table height with no table beneath it; lamp in the "
            "middle of open floor; painting placed near floor level. "
            "Before scoring, call out any specific misplacement you observe. "
            "Score 5 = all three aspects correct. "
            "Score 3 = one aspect has a clear error. "
            "Score 1 = two or more aspects have clear errors."
        ),
        "scale": 5,
    },
}

# ── Task description ───────────────────────────────────────────────────────────

DEFAULT_TASK_DESCRIPTION = (
    "Reconstruct the 3D scene shown in the TARGET reference image (a single "
    "photograph of an indoor scene), then render the reconstructed scene from "
    "the same camera viewpoint. The GENERATED image is that re-rendering. "
    "Match the TARGET on room layout, furniture identity, object placement, and "
    "object orientation. Textures, colours, materials, and lighting may "
    "legitimately differ from the TARGET — focus on geometric and structural "
    "correctness."
)

# ── Prompt templates ───────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATE = """You are evaluating a 3D scene image generated based on a specific task description. You will be shown a TARGET image of an indoor scene and a GENERATED image. Compare them, prioritising geometric structure, spatial layout, object identity, and object orientation over stylistic differences (colours, textures, artistic filters).

{criteria}

Task Description: {task_description}

Instructions:
- Use the TARGET image as the ground-truth reference for intended scene layout and camera viewpoint.
- Completely ignore colour, material, and texture differences unless they make an object unrecognisable.
- Focus your judgment on whether the GENERATED image meets the task criteria relative to the TARGET reference.

Give an integer score between 0 and {scale} (higher = better). Use the full range — do not anchor to 3-5.
First give the score; then justify in one sentence. Format: '4. The sofa correctly faces the coffee table and all objects are identifiable.'
"""

COMBINED_PROMPT_TEMPLATE = """You are evaluating a 3D scene reconstruction. You will see a TARGET image (an indoor photograph) and a GENERATED image (a render of the reconstructed scene from the same viewpoint).

Task: {task_description}

Your job: compare the two images and score the GENERATED image on each criterion below (integer 0-5, higher = better). Use the full range — do not cluster around 3-5.

IMPORTANT scoring guidance:
- Ignore colour, texture, and material differences unless they make an object unrecognisable.
- Focus on shape, position, and orientation.
- For each scored criterion, briefly state WHAT you observe in the GENERATED image before committing to a number.

Criteria:
{criteria_block}

Respond with a JSON object only — no prose outside the JSON. Use exactly this shape:
{{
  "per_object_observations": "<list every visible object in GENERATED, its expected type from TARGET, whether identity matches, and which way it faces>",
{example_block}
}}

"per_object_observations" must be filled in before the scores — this forces you to ground your scoring in specific observations rather than general impressions. Each criterion's "justification" should reference the observations above.
"""

# ── Client ─────────────────────────────────────────────────────────────────────

_GLOBAL_CLIENT: Optional[OpenAI] = None


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Return a cached OpenAI client. Falls back to ``OPENAI_API_KEY`` env var."""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or pass api_key= explicitly."
            )
        _GLOBAL_CLIENT = OpenAI(api_key=key)
    return _GLOBAL_CLIENT


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    """Base64-encode an image for inlining into a GPT message.

    OpenAI only accepts PNG / JPEG / GIF / WebP. Anything else (e.g. AVIF) is
    transcoded to PNG via PIL before encoding.
    """
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    supported = {"png", "jpg", "jpeg", "gif", "webp"}
    if ext in supported:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Transcode unsupported formats (e.g. AVIF) to PNG via PIL
    try:
        import pillow_avif  # registers AVIF support
    except ImportError:
        pass
    from io import BytesIO
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Main evaluation call ───────────────────────────────────────────────────────

def evaluate_pair(
    pred_path: str,
    target_path: str,
    task_description: str = DEFAULT_TASK_DESCRIPTION,
    criteria: Optional[dict] = None,
    model_name: str = "gpt-4o",
    client: Optional[OpenAI] = None,
) -> dict:
    """Single-call evaluation: ask GPT for all criteria at once via JSON output.

    Returns:
        {
          "<criterion_name>": {"score": float, "justification": str, "criteria": str},
          ...
          "average_score": float,
          "per_object_observations": str,
        }
    """
    if criteria is None:
        criteria = EVALUATION_CRITERIA
    if client is None:
        client = get_client()

    criteria_block = "\n".join(
        f"- {name} (0-{c['scale']}): {c['criteria']}" for name, c in criteria.items()
    )
    example_block = ",\n".join(
        f'  "{name}": {{"score": <int>, "justification": "<one sentence referencing your observations>"}}'
        for name in criteria
    )
    prompt = COMBINED_PROMPT_TEMPLATE.format(
        task_description=task_description,
        criteria_block=criteria_block,
        example_block=example_block,
    )

    results: dict = {}
    raw_text: str = ""
    error: Optional[str] = None
    try:
        target_url = f"data:image/png;base64,{encode_image(target_path)}"
        pred_url = f"data:image/png;base64,{encode_image(pred_path)}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Target image:"},
                {"type": "image_url", "image_url": {"url": target_url}},
                {"type": "text", "text": "Generated image:"},
                {"type": "image_url", "image_url": {"url": pred_url}},
            ],
        }]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        parsed = _json.loads(raw_text)
        for name in criteria:
            item = parsed.get(name, {})
            results[name] = {
                "score": float(item.get("score", 0)),
                "justification": str(item.get("justification", "missing in response")),
                "criteria": name,
            }
        results["per_object_observations"] = parsed.get("per_object_observations", "")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        for name in criteria:
            results[name] = {
                "score": 0.0,
                "justification": f"Error during evaluation: {e}",
                "criteria": name,
            }

    scores = [results[n]["score"] for n in criteria if n in results]
    results["average_score"] = sum(scores) / len(scores) if scores else 0.0
    results["_raw_response"] = raw_text
    results["_error"] = error
    return results
