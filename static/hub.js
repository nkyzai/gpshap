(function () {
  "use strict";

  const root = document.querySelector("#view-models");
  if (!root) return;

  const $ = s => document.querySelector(s);
  const esc = v => String(v ?? "").replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");

  /* ---- strip legacy listeners by cloning controls ---- */
  const ids = ["hub-search", "hub-task", "hub-sort", "hub-deep", "hub-refresh", "hub-local"];
  const values = {};
  ids.forEach(id => {
    const el = $("#" + id);
    if (!el) return;
    values[id] = el.type === "checkbox" ? el.checked : el.value;
    const clone = el.cloneNode(true);
    el.replaceWith(clone);
    if (clone.type === "checkbox") clone.checked = !!values[id];
    else clone.value = values[id] ?? "";
  });

  /* ---- banner ---- */
  let banner = $("#hub-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "hub-banner";
    const tableWrap = root.querySelector(".table-wrap");
    if (tableWrap) tableWrap.before(banner); else root.appendChild(banner);
  }
  banner.style.cssText =
    "margin:0 16px 10px;padding:10px 12px;border-radius:10px;font-size:12px;" +
    "border:1px solid var(--border);background:#111823;color:var(--muted);white-space:pre-wrap";

  function showBanner(text, kind) {
    const colors = {
      ok: "#8fe3b4", warn: "#ffcf8f", error: "#ff9fb4", info: "#9fb8e0"
    };
    banner.style.color = colors[kind] || colors.info;
    banner.style.borderColor = kind === "error" ? "#5a2f3a" : "#243044";
    banner.textContent = text;
  }

  /* ---- diagnose button ---- */
  const controls = root.querySelector(".hub-controls");
  if (controls && !$("#hub-diagnose")) {
    const b = document.createElement("button");
    b.id = "hub-diagnose";
    b.className = "secondary";
    b.textContent = "Diagnose";
    b.title = "Test Hub connectivity and show the real error";
    controls.appendChild(b);
  }

  const local = { models: [], localOnly: false, busy: false, timer: null, wasActive: false };

  const fmtBytes = b => !b ? "—" : b > 1024 ** 3
    ? (b / 1024 ** 3).toFixed(2) + " GB"
    : (b / 1024 ** 2).toFixed(0) + " MB";
  const fmtCtx = n => !n ? "—" : n >= 1000
    ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n);

  async function api(path, opts = {}) {
    opts.headers = opts.headers || {};
    if (opts.body && !(opts.body instanceof FormData)) {
      opts.headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = res.statusText;
      try { const d = await res.json(); if (d.detail) msg = d.detail; } catch (_) {}
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return res.json();
  }

  /* ---- rendering ---- */
  function render() {
    const tbody = $("#hub-table tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!local.models.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="9" class="small">
        No models to display. Type a search term and press Enter, or click "Installed only"
        to see what is already on disk.</td>`;
      tbody.appendChild(tr);
      return;
    }

    for (const m of local.models) {
      const tr = document.createElement("tr");
      tr.dataset.repo = m.id;

      const caps = (m.capabilities || []).slice(0, 7).map(c =>
        `<span class="badge ${["tools", "vision", "image-generation"].includes(c) ? "hot" : ""}
         ${c === "gated" ? "gated" : ""}">${esc(c)}</span>`).join("");

      const downloaded = m.downloaded || m.status === "ready";
      const running = m.status === "downloading" || m.status === "queued";
      const status = running
        ? `<span class="status-downloading">${esc(m.status)} ${Math.round((m.progress || 0) * 100)}%</span>
           <div class="bar"><div class="bar-fill" style="width:${(m.progress || 0) * 100}%"></div></div>`
        : m.status === "error"
          ? `<span class="status-error" title="${esc(m.error || "")}">error</span>`
          : downloaded
            ? `<span class="status-ready">installed</span>`
            : `<span class="small">not installed</span>`;

      const popularity = m.downloads
        ? (m.downloads > 1e6 ? (m.downloads / 1e6).toFixed(1) + "M"
           : m.downloads > 1e3 ? (m.downloads / 1e3).toFixed(0) + "k" : m.downloads)
        : "—";

      tr.innerHTML = `
        <td class="mid">
          <a href="${esc(m.url || ("https://huggingface.co/" + m.id))}" target="_blank"
             rel="noopener" style="color:var(--text);text-decoration:none">${esc(m.id)}</a>
          ${m.active_role ? `<br><span class="badge active">${esc(m.active_role)}</span>` : ""}
          ${m.format && m.format !== "transformers"
            ? `<br><span class="badge ${m.format === "gguf" ? "hot" : ""}">${esc(m.format)}</span>` : ""}
          ${m.note ? `<br><span class="small">${esc(m.note)}</span>` : ""}
        </td>
        <td>${esc(m.kind || "—")}</td>
        <td>${m.params_billion ? m.params_billion + "B" : "—"}</td>
        <td title="${m.size_estimated ? "Estimated from parameter count / quantisation" : "Exact file sizes from Hub"}">
          ${fmtBytes(m.local_bytes || m.size_bytes)}${m.size_estimated ? " ~" : ""}
        </td>
        <td>${fmtCtx(m.context_tokens)}</td>
        <td>${caps || "—"}</td>
        <td class="small">${esc(popularity)}${m.likes ? " · ♥" + m.likes : ""}
          ${m.last_modified ? `<br><span class="small">${esc(m.last_modified)}</span>` : ""}</td>
        <td>${status}</td>
        <td style="white-space:nowrap"></td>`;

      const actions = tr.lastElementChild;

      if (!downloaded && !running) {
        const dl = document.createElement("button");
        dl.className = "tiny";
        dl.textContent = "Download";
        dl.onclick = () => startDownload(m);
        actions.appendChild(dl);
      } else if (running) {
        const cx = document.createElement("button");
        cx.className = "tiny danger";
        cx.textContent = "Cancel";
        cx.onclick = async () => {
          try { await api("/api/models/cancel", {
            method: "POST", body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" })
          }); } catch (e) { alert(e.message); }
        };
        actions.appendChild(cx);
      } else {
        const use = document.createElement("button");
        use.className = "tiny";
        use.textContent = "Use";
        use.onclick = () => useModel(m);
        actions.appendChild(use);

        const del = document.createElement("button");
        del.className = "tiny danger";
        del.textContent = "Del";
        del.onclick = () => deleteModel(m);
        actions.appendChild(del);
      }

      const info = document.createElement("button");
      info.className = "tiny secondary";
      info.textContent = "i";
      info.title = "Full metadata: context, architecture, file list";
      info.onclick = () => showDetails(m);
      actions.appendChild(info);

      tbody.appendChild(tr);
    }
  }

  /* ---- fetching ---- */
  async function loadModels(force = false) {
    if (local.busy) return;
    local.busy = true;

    const refresh = $("#hub-refresh");
    const tbody = $("#hub-table tbody");
    if (refresh) refresh.disabled = true;

    const query = ($("#hub-search")?.value || "").trim();
    const task = $("#hub-task")?.value || "all";
    const sort = $("#hub-sort")?.value || "downloads";
    const deep = !!$("#hub-deep")?.checked;

    showBanner(local.localOnly
      ? "LOCAL INVENTORY — models already downloaded to H:\\gapshap\\models"
      : `Searching Hugging Face…  "${query || "(all models)"}"  ·  task=${task}  ·  sort=${sort}
First search can take a few seconds while context windows and file sizes are fetched.`,
      "info");

    if (tbody && !local.models.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="small">Contacting ${esc("huggingface.co")}…</td></tr>`;
    }

    try {
      let data;
      if (local.localOnly) {
        const d = await api("/api/models/local");
        data = { models: d.models, source: "local", count: d.models.length, disk: d.disk };
        showBanner(`INSTALLED LOCALLY — ${d.models.length} model(s)\n` +
          `Disk free: ${d.disk?.free_gb ?? "—"} GB · Models folder: ${d.disk?.models_gb ?? "—"} GB`, "ok");
      } else {
        const p = new URLSearchParams({
          search: query, task, sort, limit: "40",
          deep: deep ? "true" : "false", force: force ? "true" : "false"
        });
        data = await api("/api/models/browse?" + p.toString());
      }

      local.models = data.models || [];
      render();

      if ($("#disk-info")) {
        $("#disk-info").textContent =
          `disk free ${data.disk?.free_gb ?? "—"} GB · models ${data.disk?.models_gb ?? "—"} GB`;
      }

      if (data.warning) {
        showBanner(
          `LIVE SEARCH FAILED — this is NOT an empty catalogue.\n\n${data.warning}\n\n` +
          `Endpoint: ${data.endpoint}\n` +
          `Fix: Settings -> HuggingFace token (for gated repos), or add\n` +
          `set "HF_ENDPOINT=https://hf-mirror.com"\n` +
          `to run.bat if your network blocks Hugging Face. Then click Diagnose.`,
          "error");
      } else if (data.source === "empty") {
        showBanner(
          `NO MATCHES for "${data.query}" (task=${data.task}).\n` +
          `Hugging Face returned zero repositories for that exact query.\n\n` +
          `Try: qwen / llama / gemma / deepseek / phi / mistral / hermes\n` +
          `Set Task = All and sort = Recently updated for new releases.\n` +
          `Search matches repository names, not marketing names.`,
          "warn");
      } else if (data.source === "preset-only") {
        showBanner(
          `Live search returned no additional matches. Showing only built-in entries that match "${data.query}".`,
          "warn");
      } else if (data.source !== "local") {
        const gguf = local.models.filter(m => m.format === "gguf").length;
        showBanner(
          `LIVE HUB — ${data.count} results (${data.live_count ?? data.count} from the live API) ` +
          `in ${data.elapsed ?? "?"}s · ${data.endpoint}\n` +
          (gguf ? `⚠ ${gguf} GGUF result(s): these need a llama.cpp runtime, not the transformers loader.\n` : "") +
          (local.models.some(m => m.size_estimated)
            ? "Sizes marked ~ are estimates. Tick 'Exact sizes' + Refresh for real byte counts.\n" : "") +
          "Tip: sort by 'Recently updated' to see brand-new releases.",
          "ok");
      }

      pollJobs();
    } catch (e) {
      local.models = [];
      render();
      showBanner("REQUEST FAILED\n\n" + e.message +
        "\n\nIf this says 404, the /api/models/browse route is missing or duplicated in app/main.py. " +
        "Restart run.bat after editing.", "error");
    } finally {
      local.busy = false;
      if (refresh) refresh.disabled = false;
    }
  }

  /* ---- actions ---- */
  async function startDownload(modelName) {
    const downloadBtn = document.getElementById(`dl-btn-${modelName}`);
    const stopBtn = document.getElementById(`stop-dl-btn-${modelName}`);
    
    downloadBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');

    try {
        await fetch(`/api/models/download?model_name=${encodeURIComponent(modelName)}`, {
            method: 'POST'
        });
    } catch (error) {
        console.error("Download failed to start:", error);
        resetDownloadUI(modelName);
    }
}

async function cancelDownload(modelName) {
    try {
        await fetch(`/api/models/download/cancel?model_name=${encodeURIComponent(modelName)}`, {
            method: 'POST'
        });
    } catch (error) {
        console.error("Failed to cancel download:", error);
    } finally {
        resetDownloadUI(modelName);
    }
}

function resetDownloadUI(modelName) {
    const downloadBtn = document.getElementById(`dl-btn-${modelName}`);
    const stopBtn = document.getElementById(`stop-dl-btn-${modelName}`);
    if (downloadBtn) downloadBtn.classList.remove('hidden');
    if (stopBtn) stopBtn.classList.add('hidden');
}

  async function useModel(m) {
    try {
      const r = await api("/api/models/use", {
        method: "POST",
        body: JSON.stringify({ model_id: m.id, kind: m.kind })
      });
      showBanner(`ACTIVE ${r.field.toUpperCase()}: ${r.model}\n` +
        (r.llm ? `Loaded on ${r.llm.device} · ${r.llm.params_billion ?? "?"}B params · ` +
          `${r.llm.context_tokens ?? "?"} token context` : "Engine switched."), "ok");
      await loadModels(true);
      if (window.loadStatus) await window.loadStatus();
    } catch (e) {
      showBanner("COULD NOT ACTIVATE\n\n" + e.message +
        "\n\nNote: GGUF models require the llama.cpp backend (not yet enabled in this build). " +
        "Choose a safetensors / transformers repository.", "error");
    }
  }

  async function deleteModel(m) {
    if (!confirm(`Delete ${m.id} from disk (${fmtBytes(m.local_bytes || m.size_bytes)})?`)) return;
    try {
      const r = await api("/api/models/delete", {
        method: "POST", body: JSON.stringify({ model_id: m.id })
      });
      showBanner(`Deleted. Freed ${fmtBytes(r.freed_bytes)}.`, "ok");
      await loadModels(true);
    } catch (e) { showBanner(e.message, "error"); }
  }

  async function showDetails(m) {
    const box = $("#hub-detail");
    if (!box) return;
    box.classList.remove("hidden");
    box.textContent = "Fetching full metadata…";
    try {
      const d = await api("/api/models/details/" + encodeURIComponent(m.id));
      box.innerHTML = `
        <strong>${esc(d.id)}</strong> · ${esc(d.kind || "")} · ${esc(d.format || "")} · ${esc(d.library || "")}
        <div class="small">
          params ${d.params_billion ? d.params_billion + "B" : "—"} ·
          exact size ${fmtBytes(d.size_bytes)} ·
          context ${fmtCtx(d.context_tokens)} ·
          likes ${d.likes || 0} · downloads ${d.downloads || 0}
          ${d.gated ? " · GATED" : ""}
          ${d.architecture ? " · " + esc(d.architecture) : ""}
        </div>
        <div>${(d.capabilities || []).map(c => `<span class="badge">${esc(c)}</span>`).join("")}</div>
        ${d.error ? `<div class="status-error">${esc(d.error)}</div>` : ""}
        <div class="small">files: ${esc((d.files || []).slice(0, 30).join(", "))}</div>`;
    } catch (e) { box.textContent = e.message; }
  }

  /* ---- live download progress ---- */
  async function pollJobs(forceStart = false) {
    try {
      const d = await api("/api/models/jobs");
      const active = (d.jobs || []).filter(j => ["queued", "downloading"].includes(j.status));

      for (const j of d.jobs) {
        const row = document.querySelector(`#hub-table tbody tr[data-repo="${j.repo_id}"]`);
        if (!row) continue;
        const cell = row.children[7];
        if (!cell) continue;
        if (["queued", "downloading"].includes(j.status)) {
          cell.innerHTML = `<span class="status-downloading">${esc(j.status)} ${Math.round(j.progress * 100)}%</span>
            <div class="bar"><div class="bar-fill" style="width:${j.progress * 100}%"></div></div>
            <div class="small">${fmtBytes(j.downloaded)} / ${fmtBytes(j.total)}</div>`;
        } else if (j.status === "ready") {
          cell.innerHTML = `<span class="status-ready">installed</span>`;
        } else if (j.status === "error") {
          cell.innerHTML = `<span class="status-error" title="${esc(j.error)}">error</span>`;
        } else if (j.status === "cancelled") {
          cell.innerHTML = `<span class="small">cancelled</span>`;
        }
      }

      const prog = $("#hub-progress");
      if (prog) {
        prog.innerHTML = active.map(j =>
          `<div class="small">${esc(j.repo_id)} — ${esc(j.status)} ${Math.round(j.progress * 100)}%
           (${fmtBytes(j.downloaded)} / ${fmtBytes(j.total)}) ${esc(j.stage || "")}
           <div class="bar"><div class="bar-fill" style="width:${j.progress * 100}%"></div></div></div>`
        ).join("");
      }

      if (active.length || forceStart) {
        if (!local.timer) {
          local.timer = setInterval(async () => {
            if (document.hidden) return;
            try {
              const data = await api("/api/models/jobs");
              const still = data.jobs.filter(j => ["queued", "downloading"].includes(j.status));

              for (const j of data.jobs) {
                const row = document.querySelector(`#hub-table tbody tr[data-repo="${j.repo_id}"]`);
                if (!row) continue;
                const cell = row.children[7];
                if (!cell) continue;
                if (["queued", "downloading"].includes(j.status)) {
                  cell.innerHTML = `<span class="status-downloading">${Math.round(j.progress * 100)}%</span>
                    <div class="bar"><div class="bar-fill" style="width:${j.progress * 100}%"></div></div>`;
                }
              }

              if (!still.length) {
                clearInterval(local.timer);
                local.timer = null;
                await loadModels(true);
                if (window.loadStatus) await window.loadStatus();
                showBanner("Download finished. Click Use to activate the model.", "ok");
              }
            } catch (_) {
              clearInterval(local.timer);
              local.timer = null;
            }
          }, 1500);
        }
      }
    } catch (_) {}
  }

  async function diagnose() {
    showBanner("Running Hub diagnostics…", "info");
    try {
      const d = await api("/api/hub/diagnose");
      const lines = [
        `ENDPOINT: ${d.endpoint}`,
        `PYTHON: ${d.python} · HF TOKEN: ${d.token_set ? "set" : "not set"}`,
        `VERSIONS: ${Object.entries(d.versions || {}).map(([k, v]) => k + "=" + v).join(", ")}`,
        "",
        ...(d.checks || []).map(c => c.ok
          ? `OK   ${c.check}: ${c.count} results in ${c.seconds}s -> ${(c.samples || []).join(", ")}`
          : `FAIL ${c.check}: ${c.error}`),
        "",
        "ADVICE:",
        ...(d.advice || []).map(a => "• " + a)
      ];
      showBanner(lines.join("\n"), d.checks?.every(c => c.ok) ? "ok" : "error");
    } catch (e) { showBanner("Diagnose failed: " + e.message, "error"); }
  }

  /* ---- publish replacements so app.js calls the fixed versions ---- */
  window.loadModels = loadModels;
  window.renderModels = render;
  window.startDownload = startDownload;
  window.useModel = useModel;
  window.showDetails = showDetails;
  window.watchProgress = () => pollJobs(true);
  window.gapshapHubRefresh = () => loadModels(true);

  /* ---- bindings ---- */
  let debounce = null;
  const search = $("#hub-search");
  if (search) {
    search.addEventListener("input", () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => loadModels(false), 500);
    });
    search.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        e.preventDefault();
        clearTimeout(debounce);
        local.localOnly = false;
        const b = $("#hub-local");
        if (b) b.textContent = "Installed only";
        loadModels(true);
      }
    });
  }



  $("#hub-refresh")?.addEventListener("click", () => loadModels(true));
  $("#hub-diagnose")?.addEventListener("click", diagnose);

  $("#hub-task")?.addEventListener("change", () => loadModels(true));
  $("#hub-deep")?.addEventListener("change", () => loadModels(true));

  $("#hub-sort")?.addEventListener("change", () => {
    if ($("#hub-sort").value === "size" && $("#hub-deep")) {
      $("#hub-deep").checked = true; // size sorting needs real sizes
    }
    loadModels(true);
  });

  $("#hub-local")?.addEventListener("click", () => {
    local.localOnly = !local.localOnly;
    $("#hub-local").textContent = local.localOnly ? "Browse Hub" : "Installed only";
    loadModels(true);
  });

  /* When the user opens the Models tab, app.js calls window.loadModels(). */
  showBanner(
    "MODEL HUB READY\n" +
    "Type a repository name (qwen, llama, gemma, deepseek, phi, hermes) and press Enter.\n" +
    "Task = All + sort = Recently updated finds the newest releases.\n" +
    "Click Diagnose if the table stays empty.",
    "info");
})();

