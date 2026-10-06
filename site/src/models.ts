// Fixed colour per model (entity), never per rank: a model keeps its colour on every view.
// Slots follow the validated categorical palette (dataviz reference palette, slots 1–6),
// checked with validate_palette.js in light and dark mode. Light-mode contrast is below 3:1
// for some slots, so every chart ships a legend with text labels and a data table.
import { prefersDark } from "./theme";

const LIGHT: Record<string, string> = {
  dead_reckoning: "#2a78d6",
  kf_cv: "#eb6834",
  imm: "#1baf7a",
  knn_route: "#eda100",
  gru: "#e87ba4",
  gru_mdn: "#008300",
};
const DARK: Record<string, string> = {
  dead_reckoning: "#3987e5",
  kf_cv: "#d95926",
  imm: "#199e70",
  knn_route: "#c98500",
  gru: "#d55181",
  gru_mdn: "#008300",
};

export const MODEL_ORDER = Object.keys(LIGHT);

export function modelColor(id: string): string {
  return (prefersDark() ? DARK : LIGHT)[id] ?? "#888888";
}
