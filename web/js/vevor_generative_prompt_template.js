import { app } from "../../../scripts/app.js";

/** Comfy 1.48 有时未把 widgets_values_named.prompt_skeleton 灌进多行框；configure 后强制同步。 */
function syncPromptSkeletonWidget(node, info) {
  if (!node?.widgets || !info) return;
  const skeleton =
    info.widgets_values_named?.prompt_skeleton ??
    (Array.isArray(info.widgets_values) ? info.widgets_values[1] : undefined);
  if (skeleton == null) return;
  const w = node.widgets.find((x) => x.name === "prompt_skeleton");
  if (!w) return;
  w.dynamicPrompts = false;
  const text = String(skeleton);
  if (String(w.value) !== text) {
    w.value = text;
  }
  queueMicrotask(() => {
    if (w.inputEl) {
      w.inputEl.style.minHeight = "280px";
    }
  });
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
      syncPromptSkeletonWidget(this, info);
    };
  },
});
