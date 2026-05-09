// Pings the HF Space's runtime API to set a status badge on the demo button.
// Handles three cases:
//   - Space exists and is up/asleep/down  -> green/yellow/red badge
//   - Space does not exist yet (404)      -> "demo not yet deployed" note
//   - CORS / network error                -> silent (button text already says "coming soon")

(async () => {
  const SPACE = "idonithid/SONAR-demo";
  const el = document.getElementById("space-status");
  const link = document.getElementById("demo-link");
  if (!el) return;

  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${SPACE}/runtime`, {
      cache: "no-store",
    });
    if (r.status === 404) {
      el.textContent = "Demo Space is not yet deployed. The findings and figures on this page work without it.";
      return;
    }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const stage = (j && j.stage) || "unknown";

    if (link) {
      link.classList.remove("disabled");
      link.removeAttribute("aria-disabled");
      link.href = `https://huggingface.co/spaces/${SPACE}`;
      link.textContent = "Try the live demo →";
    }

    if (stage === "RUNNING") {
      el.textContent = "🟢 Demo is online — click above to try it.";
      el.classList.add("up");
    } else if (stage === "SLEEPING") {
      el.textContent = "🟡 Demo is asleep — first click will warm it up (~30–60 s).";
      el.classList.add("warm");
    } else if (stage === "STOPPED" || stage === "BUILD_ERROR" || stage === "RUNTIME_ERROR") {
      el.textContent = `🔴 Demo is offline (${stage.toLowerCase()}). The figures and findings on this page still work.`;
      el.classList.add("down");
    } else {
      el.textContent = `Demo state: ${stage.toLowerCase()}`;
    }
  } catch (e) {
    /* silent: button already says "coming soon" */
  }
})();
