// Pings the HF Space's runtime API to set a green/yellow/red status badge.
// Falls back gracefully if CORS blocks the request — the user can still click.

(async () => {
  const SPACE = "idonithid/SONAR-demo";
  const el = document.getElementById("space-status");
  const link = document.getElementById("demo-link");
  if (!el) return;

  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${SPACE}/runtime`, {
      cache: "no-store",
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const stage = (j && j.stage) || "unknown";

    if (stage === "RUNNING") {
      el.textContent = "🟢 Demo is online — click above to try it.";
      el.classList.add("up");
    } else if (stage === "SLEEPING") {
      el.textContent = "🟡 Demo is asleep — first click will warm it up (~30–60 s).";
      el.classList.add("warm");
    } else if (stage === "STOPPED" || stage === "BUILD_ERROR" || stage === "RUNTIME_ERROR") {
      el.textContent = `🔴 Demo is offline (${stage.toLowerCase()}). The figures and findings on this page still work.`;
      el.classList.add("down");
      if (link) link.style.opacity = "0.55";
    } else {
      el.textContent = `Demo state: ${stage.toLowerCase()}`;
    }
  } catch (e) {
    el.textContent = "Demo status unknown — click above to check directly.";
  }
})();
