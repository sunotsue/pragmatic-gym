// Screenshot every Pragmatic Gym view at 1440×900 (and optionally full page).
//   node scripts/screenshot_app.mjs [outDir] [--interact]
import { createRequire } from "module";
const require = createRequire(import.meta.url + "/../../apps/web/");
const { chromium } = require("playwright");

const out = process.argv[2] || "docs/design/app";
const interact = process.argv.includes("--interact");
const base = process.env.PGYM_WEB ?? "http://localhost:3000";
const routes = [["home", "/"], ["explorer", "/explorer"], ["counterfactual", "/counterfactual"], ["playground", "/playground"], ["data", "/data"], ["results", "/results"]];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
for (const [name, path] of routes) {
  await page.goto(base + path, { waitUntil: "load" });
  await page.waitForTimeout(3500);
  if (interact) {
    const click = async (text, wait = 8000) => { const b = page.getByRole("button", { name: text }).first(); if (await b.count()) { await b.click(); await page.waitForTimeout(wait); } };
    if (name === "explorer") await click("Run model", 12000);
    if (name === "counterfactual") await click(/Run experiment/, 25000);
    if (name === "playground") { await click("Let the model act", 6000); await click("Reveal hidden intent", 500); }
  }
  await page.screenshot({ path: `${out}/${name}-1440.png` });
  await page.screenshot({ path: `${out}/${name}-full.png`, fullPage: true });
  console.log("shot", name);
}
await browser.close();
