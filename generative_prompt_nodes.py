"""Canvas nodes: variable maps, catalog prompt template (P0–P3), space budget, review apply."""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generative_pipeline import (  # noqa: E402
    _blocks_as_list,
    _parse_translation_map,
    build_analyze_variable_map,
    build_generate_variable_map,
    build_review_variable_map,
    build_space_section,
    build_translate_variable_map,
    render_prompt_from_catalog,
)
from generative_template import (  # noqa: E402
    PROMPT_KEYS,
    parse_variables_json,
    resolve_template,
)


def _vars_to_json(variables: dict) -> str:
    return json.dumps(variables, ensure_ascii=False)


class VevorGenerativePromptTemplate:
    """Render catalog ``{{placeholder}}`` prompt; P2 配置用 ``template_override``。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_key": (list(PROMPT_KEYS),),
                "prompt_skeleton": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "提示词骨架（catalog + 占位符；只读参考，Queue 仍走 variables_json）",
                    },
                ),
                "variables_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "forceInput": True,
                    },
                ),
            },
            "optional": {
                "template_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "placeholder": "P2 覆写（留空=包内 catalog）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "render"
    CATEGORY = "vevor/generative"
    OUTPUT_NODE = True

    def render(
        self,
        prompt_key: str,
        prompt_skeleton: str,
        variables_json: str,
        template_override: str = "",
    ):
        # 画布只读展示；执行仍以 resolve_template + variables 为准
        _ = prompt_skeleton or resolve_template(prompt_key, "")
        variables = parse_variables_json(variables_json)
        prompt = render_prompt_from_catalog(
            prompt_key,
            variables,
            str(template_override or ""),
        )
        preview = prompt if len(prompt) <= 12000 else prompt[:12000] + "\n…(truncated)"
        return {"ui": {"text": [preview]}, "result": (prompt,)}


class VevorGenerativeVariablesJsonPatch:
    """在 ``variables_json`` 上追加键值；可链式串联（本节点输出 → 下一节点 variables_json）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variables_json": (
                    "STRING",
                    {"forceInput": True, "multiline": False},
                ),
                "patch_key": (
                    "STRING",
                    {
                        "default": "extraSection",
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "patch_value": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("variables_json",)
    FUNCTION = "patch"
    CATEGORY = "vevor/generative"

    def patch(
        self,
        variables_json: str,
        patch_key: str,
        patch_value: str = "",
    ):
        data = parse_variables_json(variables_json)
        key = str(patch_key or "").strip()
        if key:
            val = patch_value
            if val is not None and str(val).strip():
                data[key] = val if isinstance(val, str) else str(val)
        return (_vars_to_json(data),)


class VevorGenerativeShowPrompt:
    """OUTPUT 节点：展示渲染后的 prompt（与 TranslationReport 同 ui.text 机制）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "show"
    CATEGORY = "vevor/generative"
    OUTPUT_NODE = True

    def show(self, prompt: str):
        text = str(prompt or "")
        preview = text if len(text) <= 16000 else text[:16000] + "\n…(truncated)"
        return {"ui": {"text": [preview]}, "result": (text,)}


class VevorGenerativeAnalyzeVars:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schema_template": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "留空=generative_analyze_prompt.txt",
                    },
                ),
                "user_line": (
                    "STRING",
                    {
                        "default": "识别这张图上所有可见文字块，只输出 JSON 数组。",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("variables_json",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, schema_template: str, user_line: str):
        vars_map = build_analyze_variable_map(schema_template, user_line)
        return (_vars_to_json(vars_map),)


class VevorGenerativeSpaceBudget:
    """Expose ``{{spaceSection}}`` for Translate prompt (design-agent space budget)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "lang_code": (["de", "fr", "it", "es"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("space_section",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, blocks, lang_code: str):
        _ = lang_code
        section = build_space_section(blocks)
        return (section,)


class VevorGenerativeTranslateVars:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "target_lang": ("STRING", {"default": "德语"}),
                "lang_code": (["de", "fr", "it", "es"],),
            },
            "optional": {
                "lexicon_extra": ("STRING", {"default": "", "forceInput": True}),
                "retry_variables_json": ("STRING", {"default": "", "forceInput": True}),
                "space_section": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("variables_json",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(
        self,
        blocks,
        target_lang: str,
        lang_code: str,
        lexicon_extra: str = "",
        retry_variables_json: str = "",
        space_section: str = "",
    ):
        vars_map = build_translate_variable_map(
            blocks,
            target_lang,
            lang_code,
            lexicon_extra=str(lexicon_extra or ""),
            retry_json=str(retry_variables_json or ""),
        )
        override = str(space_section or "").strip()
        if override:
            vars_map["spaceSection"] = override
        return (_vars_to_json(vars_map),)


class VevorGenerativeGenerateVars:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "target_lang": ("STRING", {"default": "德语"}),
                "font_dir": ("STRING", {"default": "fonts"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("variables_json",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, blocks, target_lang: str, font_dir: str):
        vars_map = build_generate_variable_map(blocks, target_lang, font_dir)
        return (_vars_to_json(vars_map),)


class VevorGenerativeReviewVars:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "target_lang": ("STRING", {"default": "德语"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("variables_json",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, blocks, target_lang: str):
        vars_map = build_review_variable_map(blocks, target_lang)
        return (_vars_to_json(vars_map),)


class VevorGenerativeApplyReview:
    """Merge review LLM JSON (partial fixes) into ``blocks[].translation``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "review_response": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("blocks",)
    FUNCTION = "apply"
    CATEGORY = "vevor/generative"

    def apply(self, blocks, review_response: str):
        data = [dict(b) for b in _blocks_as_list(blocks)]
        mapping = _parse_translation_map(review_response)
        for b in data:
            tr = mapping.get(int(b.get("id", 0)))
            if tr:
                b["translation"] = tr
        return (data,)


NODE_CLASS_MAPPINGS = {
    "VevorGenerativePromptTemplate": VevorGenerativePromptTemplate,
    "VevorGenerativeVariablesJsonPatch": VevorGenerativeVariablesJsonPatch,
    "VevorGenerativeShowPrompt": VevorGenerativeShowPrompt,
    "VevorGenerativeAnalyzeVars": VevorGenerativeAnalyzeVars,
    "VevorGenerativeSpaceBudget": VevorGenerativeSpaceBudget,
    "VevorGenerativeTranslateVars": VevorGenerativeTranslateVars,
    "VevorGenerativeGenerateVars": VevorGenerativeGenerateVars,
    "VevorGenerativeReviewVars": VevorGenerativeReviewVars,
    "VevorGenerativeApplyReview": VevorGenerativeApplyReview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorGenerativePromptTemplate": "Generative · Prompt Template",
    "VevorGenerativeVariablesJsonPatch": "Generative · Variables JSON Patch",
    "VevorGenerativeShowPrompt": "Generative · Show Rendered Prompt",
    "VevorGenerativeAnalyzeVars": "Generative · Analyze Variables",
    "VevorGenerativeSpaceBudget": "Generative · Space Budget",
    "VevorGenerativeTranslateVars": "Generative · Translate Variables",
    "VevorGenerativeGenerateVars": "Generative · Generate Variables",
    "VevorGenerativeReviewVars": "Generative · Review Variables",
    "VevorGenerativeApplyReview": "Generative · Apply Review",
}
