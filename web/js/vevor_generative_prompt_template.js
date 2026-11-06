import { app } from "../../../scripts/app.js";

/** Comfy 1.48 有时未把 widgets_values_named.prompt_skeleton 灌进多行框；configure 后强制同步。 */
function syncWidgetFromNamed(node, info, name, minHeightPx) {
  if (!node?.widgets || !info) return;
  const named = info.widgets_values_named ?? {};
  let value = named[name];
  if (value == null && Array.isArray(info.widgets_values)) {
    const wv = info.widgets_values;
    if (name === "prompt_skeleton") {
      if (wv.length === 1) value = wv[0];
      else if (wv.length >= 2 && typeof wv[0] === "string") {
        const legacyKeys = ["analyze", "translate", "review", "image"];
        value = legacyKeys.includes(wv[0]) ? wv[1] : wv[0];
      }
    }
  }
  if (value == null) return;
  const w = node.widgets.find((x) => x.name === name);
  if (!w) return;
  w.dynamicPrompts = false;
  const text = String(value);
  if (String(w.value) !== text) {
    w.value = text;
  }
  if (minHeightPx && w.inputEl) {
    queueMicrotask(() => {
      w.inputEl.style.minHeight = `${minHeightPx}px`;
    });
  }
}

function syncPromptTemplateWidgets(node, info) {
  const sk = String(
    (info.widgets_values_named ?? {}).prompt_skeleton ??
      info.widgets_values?.[0] ??
      "",
  );
  const isAnalyze =
    /analyze/i.test(String(info.title ?? "")) ||
    /PromptTemplate_analyze/i.test(
      String(info.properties?.["Node name for S&R"] ?? ""),
    ) ||
    sk.length > 800;
  syncWidgetFromNamed(node, info, "prompt_skeleton", isAnalyze ? 360 : 120);
}

app.registerExtension({
  name: "Vevor.GenerativePromptTemplate",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "VevorGenerativePromptTemplate") return;
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      if (origConfigure) {
        origConfigure.call(this, info);
      }
      syncPromptTemplateWidgets(this, info);
    };
  },
});
