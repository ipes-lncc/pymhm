/* Enlarge scientific figures without changing their numerical color scales. */
(() => {
  const install = () => {
    let dialog = document.querySelector(".figure-dialog");
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.className = "figure-dialog";
      dialog.setAttribute("aria-label", "Scientific figure");
      const figure = document.createElement("img");
      const footer = document.createElement("footer");
      const caption = document.createElement("span");
      const original = document.createElement("a");
      original.textContent = "Open original";
      original.target = "_blank";
      original.rel = "noopener";
      const close = document.createElement("button");
      close.textContent = "Close";
      close.addEventListener("click", () => dialog.close());
      footer.append(caption, original, close);
      dialog.append(figure, footer);
      dialog.addEventListener("click", event => {
        if (event.target === dialog) dialog.close();
      });
      document.body.append(dialog);
    }
    document.querySelectorAll(
      ".md-content img[src*='figures/'], .md-content img[src*='assets/tutorials/'], " +
      ".md-content img[src$='assets/architecture.svg']"
    ).forEach(image => {
      if (image.dataset.scientificFigure) return;
      image.dataset.scientificFigure = "true";
      image.tabIndex = 0;
      image.setAttribute("role", "button");
      image.setAttribute("aria-label", `Enlarge: ${image.alt}`);
      const open = () => {
        const enlarged = dialog.querySelector("img");
        enlarged.src = image.src;
        enlarged.alt = image.alt;
        dialog.querySelector("span").textContent = image.alt;
        dialog.querySelector("a").href = image.src;
        dialog.showModal();
      };
      image.addEventListener("click", event => {
        event.preventDefault();
        open();
      });
      image.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open();
        }
      });
    });
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
})();
