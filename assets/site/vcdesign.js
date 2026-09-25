const regimes = {
  measured: {
    eyebrow: "Measured-response regime",
    title: "Use the observed candidate effect directly.",
    copy: "When candidate responses have already been profiled, response-profile retrieval is the more direct source of evidence for alignment with the requested transition.",
    flow: ["candidate response", "target alignment", "ranked candidates"]
  },
  unseen: {
    eyebrow: "Response-unseen regime",
    title: "Predict the candidate effect from static knowledge.",
    copy: "CED learns from historical perturbation responses, STRING, and MAP-KG, then scores a new candidate without using that candidate's own measured response at deployment.",
    flow: ["candidate knowledge", "predicted effect", "ranked candidates"]
  }
};

function renderRegime(key) {
  const regime = regimes[key];
  if (!regime) return;
  document.querySelector("#regime-eyebrow").textContent = regime.eyebrow;
  document.querySelector("#regime-title").textContent = regime.title;
  document.querySelector("#regime-copy").textContent = regime.copy;
  document.querySelector("#regime-flow").innerHTML = regime.flow
    .map((item, index) => `${index ? "<i>→</i>" : ""}<span>${item}</span>`)
    .join("");
}

document.querySelectorAll(".regime-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".regime-tab").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    renderRegime(button.dataset.regime);
  });
});

const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector("#site-nav");
menuButton?.addEventListener("click", () => {
  const open = nav.classList.toggle("is-open");
  menuButton.setAttribute("aria-expanded", String(open));
});
nav?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  nav.classList.remove("is-open");
  menuButton?.setAttribute("aria-expanded", "false");
}));

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.querySelector(`#${button.dataset.copyTarget}`);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.innerText);
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = button.dataset.copyTarget === "download-command" ? "Copy command" : "Copy"; }, 1600);
    } catch {
      button.textContent = "Select code";
    }
  });
});
