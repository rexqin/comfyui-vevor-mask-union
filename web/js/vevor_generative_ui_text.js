import { app } from "../../../scripts/app.js";

/** 与 VevorGenerativeShowPrompt / PromptTemplate 的 Python ui.text 对齐 */
const UI_TEXT_NODE_CLASSES = new Set([
  "VevorGenerativeShowPrompt",
  "VevorGenerativePromptTemplate",
]);

const PREVIEW_WIDGET = "rendered_prompt_preview";

function textFromUiPayload(message) {
  if (!message) return "";
  const t = message.text;
  if (t == null) return "";
  if (Array.isArray(t)) return t.map(String).join("");
  return String(t);
}

function populateRenderedText(node, text) {
  const body = String(text ?? "");
  let w = node.widgets?.find((x) => x.name === PREVIEW_WIDGET);
  if (!w) {
    w = node.addWidget("text", PREVIEW_WIDGET, body, () => {}, {
      multiline: true,
    });
    if (w.inputEl) {
      w.inputEl.readOnly = true;
      w.inputEl.style.minHeight = "120px";
      w.inputEl.style.opacity = "0.88";
    }
  } else {
    w.value = body;
  }
  requestAnimationFrame(() => {
    node.computeSize?.();
    node.onResize?.(node.size);
    app.graph?.setDirtyCanvas?.(true, true);
  });
}

function installUiTextHooks(nodeType, nodeData) {
  if (!UI_TEXT_NODE_CLASSES.has(nodeData.name)) return;

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    populateRenderedText(this, "");
    return r;
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    populateRenderedText(this, textFromUiPayload(message));
    try {
      onExecuted?.apply(this, arguments);
    } catch {
      /* 其它扩展的 onExecuted 可能误处理非 audio 等 ui 载荷 */
    }
  };
}

app.registerExtension({
  name: "Vevor.GenerativeUiText",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    installUiTextHooks(nodeType, nodeData);
  },
});
