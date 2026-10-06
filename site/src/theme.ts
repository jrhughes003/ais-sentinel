/** True when the page is (forced or OS-) dark. Kept out of map.ts so light views stay light. */
export function prefersDark(): boolean {
  const forced = document.documentElement.dataset.theme;
  if (forced) return forced === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
