const state = {
  user: null,
  projects: [],
  currentProject: null,
  files: [],
  messages: [],
  tab: "chat",
  models: [],
  localOnly: false,
  recognizing: false,
  recognition: null,
  streaming: false,
  pendingFileIds: [],
  theme: localStorage.getItem("gpshap-theme") || "dark",
  showArchived: false,
  lastUserMessage: "",
  localModels: [],
};
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const esc = (v) => String(v ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#39;");

function md(text) {
  let h = esc(text);
  h = h.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, l, c) => `<pre><code>${c.replace(/\n$/, "")}</code></pre>`);
  h = h.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/(^|\s)\*([^*\n]+)\*/g, "$1<em>$2</em>");
  h = h.replace(/^#{3}\s?(.*)$/gm, "<h4>$1</h4>").replace(/^#{2}\s?(.*)$/gm, "<h3>$1</h3>");
  h = h.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/^\s*[-*]\s+(.*)$/gm, "<li>$1</li>");
  h = h.split(/\n{2,}/).map((p) => {
    if (p.startsWith("<h") || p.startsWith("<pre") || p.startsWith("<li")) return p;
    return `<p>${p.replace(/\n/g, "<br>")}</p>`;
  }).join("");
  return h;
}

async function api(path, opts = {}) {
  opts.headers = opts.headers || {};
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, opts);
  if (res.status === 401) {
    state.user = null;
    renderAuthOnly();
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const raw = await res.text();
    let msg = raw || res.statusText;
    try {
      const parsed = JSON.parse(raw);
      if (parsed.detail) msg = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

async function ssePost(url, body, on) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  await readSSE(res, on);
}

async function readSSE(res, on) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let ev = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        try { on(ev, JSON.parse(data)); }
        catch (_) { on(ev, { raw: data }); }
      }
    }
  }
}

function downloadJSON(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------- theme ---------------- */
const THEMES = ["dark", "midnight", "light", "contrast"];
function applyTheme(theme) {
  if (!THEMES.includes(theme)) theme = "dark";
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("gpshap-theme", theme);
  const sel = $("#set-theme");
  if (sel) sel.value = theme;
}
function cycleTheme() {
  const idx = THEMES.indexOf(state.theme);
  applyTheme(THEMES[(idx + 1) % THEMES.length]);
}

/* ---------------- auth / boot ---------------- */
function renderAuthOnly() {
  $("#auth-panel").classList.remove("hidden");
  $("#project-panel").classList.add("hidden");
  $("#current-project").textContent = "Not signed in";
}
function renderApp() {
  $("#auth-panel").classList.add("hidden");
  $("#project-panel").classList.remove("hidden");
}
async function boot() {
  applyTheme(state.theme);
  bind();
  setupVoice();
  try {
    const d = await api("/api/me");
    state.user = d.user;
    await Promise.all([
      loadProjects(),
      loadStatus(),
      loadModelSelect(),
      loadResources(),
      loadUsage(),
    ]);
    renderApp();
    switchTab("chat");
    startPolling();
  } catch (_) {
    renderAuthOnly();
  }
}
async function doAuth(path) {
  const username = $("#auth-username").value.trim();
  const password = $("#auth-password").value;
  if (!username || !password) return alert("Enter username and password.");
  try {
    const d = await api(path, {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    state.user = d.user;
    await Promise.all([
      loadProjects(),
      loadStatus(),
      loadModelSelect(),
      loadResources(),
      loadUsage(),
    ]);
    renderApp();
    switchTab("chat");
    startPolling();
  } catch (e) {
    alert(e.message);
  }
}
async function loadStatus() {
  try {
    const s = await api("/api/status");
    $("#engine-badge").textContent =
      `engine: ${s.llm.loaded ? "LLM ready" : "loading…"} · ${s.llm.device}` +
      `${s.llm.cuda_available ? " · " + (s.llm.gpu_name || "GPU") : ""}`;
    $("#model-chip").textContent = (s.llm.model_id || "").split("/").pop();
    $("#studio-model").textContent = s.imagegen.model_id || "";
    $("#img-warn").textContent = s.imagegen.cpu_warning
      ? "⚠ CPU diffusion is very slow - lower steps or use a GPU."
      : "";
    $("#status-box").textContent =
      `LLM      : ${s.llm.model_id}\n` +
      `device   : ${s.llm.device}${s.llm.dtype ? " · " + s.llm.dtype : ""}\n` +
      `params   : ${s.llm.params_billion ? s.llm.params_billion + "B" : "-"}\n` +
      `context  : ${s.llm.context_tokens || "-"} tok\n` +
      (s.llm.vram_allocated_mb ? `vram     : ${s.llm.vram_allocated_mb} MB\n` : "") +
      `vision   : ${s.vision.model_id} ${s.vision.downloaded ? "(ready)" : "(not downloaded)"}\n` +
      `embed    : ${s.embedding.model}\n` +
      `image    : ${s.imagegen.model_id}\n` +
      `disk free: ${s.disk.free_gb ?? "-"} GB (models ${s.disk.models_gb ?? "-"} GB)`;
  } catch (_) {}
}

/* ---------------- tabs ---------------- */
function switchTab(tab) {
  state.tab = tab;
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  ["chat", "models", "studio", "settings"].forEach((t) => {
    $("#view-" + t).classList.toggle("hidden", t !== tab);
  });
  $("#side-chat").classList.toggle("hidden", tab !== "chat");
  $("#side-status").classList.toggle("hidden", tab === "chat");
  if (tab === "models") loadModels();
  if (tab === "studio") loadGallery();
  if (tab === "settings") fillSettings();
}

/* ---------------- context menu ---------------- */
function openMenu(x, y, items) {
  const menu = $("#context-menu");
  menu.innerHTML = "";
  for (const item of items) {
    if (item.sep) {
      menu.insertAdjacentHTML("beforeend", "<hr>");
      continue;
    }
    const b = document.createElement("button");
    b.textContent = item.label;
    if (item.danger) b.className = "danger";
    b.onclick = () => {
      closeMenu();
      item.onClick();
    };
    menu.appendChild(b);
  }
  menu.classList.remove("hidden");
  requestAnimationFrame(() => {
    const rect = menu.getBoundingClientRect();
    menu.style.left = Math.min(x, window.innerWidth - rect.width - 10) + "px";
    menu.style.top = Math.min(y, window.innerHeight - rect.height - 10) + "px";
  });
}
function closeMenu() {
  $("#context-menu").classList.add("hidden");
}

/* ---------------- projects ---------------- */
async function loadProjects() {
  const d = await api("/api/projects/v2");
  state.projects = d.projects;
  renderProjects();
  if (!state.currentProject && state.projects.length) {
    await selectProject(state.projects[0].id);
  }
  await loadUsage();
}
function currentProject() {
  return state.projects.find((p) => p.id === state.currentProject) || null;
}
function renderProjects() {
  const c = $("#project-list");
  c.innerHTML = "";
  const list = state.projects.filter((p) => state.showArchived || !p.archived);
  if (!list.length) {
    c.innerHTML = '<div class="small">No projects yet.</div>';
    return;
  }
  for (const p of list) {
    const el = document.createElement("div");
    el.className = "project-item" +
      (state.currentProject === p.id ? " active" : "") +
      (p.archived ? " archived" : "");
    el.draggable = true;
    el.dataset.id = String(p.id);
    const name = document.createElement("span");
    name.className = "p-name";
    name.textContent = (p.pinned ? "📌 " : "") + p.name;
    const menuBtn = document.createElement("span");
    menuBtn.className = "p-menu";
    menuBtn.textContent = "⋮";
    menuBtn.title = "Project options";
    menuBtn.onclick = (e) => {
      e.stopPropagation();
      openProjectMenu(e.clientX, e.clientY, p);
    };
    el.onclick = () => selectProject(p.id);
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      openProjectMenu(e.clientX, e.clientY, p);
    });
    el.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", String(p.id));
    });
    el.addEventListener("dragover", (e) => e.preventDefault());
    el.addEventListener("drop", async (e) => {
      e.preventDefault();
      const from = Number(e.dataTransfer.getData("text/plain"));
      const to = p.id;
      if (!from || from === to) return;
      const ids = state.projects.map((x) => x.id);
      const fromIdx = ids.indexOf(from);
      const toIdx = ids.indexOf(to);
      if (fromIdx === -1 || toIdx === -1) return;
      ids.splice(fromIdx, 1);
      ids.splice(toIdx, 0, from);
      await reorderProjects(ids);
    });
    el.append(name, menuBtn);
    c.appendChild(el);
  }
}
function openProjectMenu(x, y, project) {
  openMenu(x, y, [
    { label: "Open", onClick: () => selectProject(project.id) },
    { label: "Rename", onClick: () => renameProject(project) },
    { label: "Duplicate", onClick: () => duplicateProject(project) },
    { label: "Export", onClick: () => exportProject(project) },
    { sep: true },
    {
      label: project.pinned ? "Unpin" : "Pin",
      onClick: () => patchProject(project.id, { pinned: !project.pinned }),
    },
    {
      label: project.archived ? "Unarchive" : "Archive",
      onClick: () => patchProject(project.id, { archived: !project.archived }),
    },
    { sep: true },
    { label: "Move up", onClick: () => moveProject(project.id, -1) },
    { label: "Move down", onClick: () => moveProject(project.id, 1) },
    { sep: true },
    { label: "Clear chat", onClick: () => clearProject(project) },
    { label: "Delete", danger: true, onClick: () => deleteProject(project) },
  ]);
}
async function createProject() {
  const name = prompt("Project name:");
  if (!name || !name.trim()) return;
  const sys = prompt("Optional system prompt:", "") || "";
  try {
    await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: name.trim(), system_prompt: sys }),
    });
    await loadProjects();
  } catch (e) {
    alert(e.message);
  }
}
async function selectProject(id) {
  state.currentProject = id;
  state.pendingFileIds = [];
  renderProjects();
  const p = currentProject();
  $("#current-project").textContent = p ? p.name : "Project";
  await Promise.all([loadMessages(), loadFiles(), loadUsage()]);
}
async function patchProject(projectId, patch) {
  try {
    await api(`/api/projects/${projectId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    await loadProjects();
  } catch (e) {
    alert(e.message);
  }
}
async function renameProject(project) {
  const name = prompt("Rename project:", project.name);
  if (!name || !name.trim()) return;
  await patchProject(project.id, { name: name.trim() });
}
async function duplicateProject(project) {
  try {
    await api(`/api/projects/${project.id}/duplicate`, { method: "POST" });
    await loadProjects();
  } catch (e) {
    alert(e.message);
  }
}
async function exportProject(project) {
  try {
    const data = await api(`/api/projects/${project.id}/export`);
    downloadJSON(`gpshap_project_${project.id}.json`, data);
  } catch (e) {
    alert(e.message);
  }
}
async function deleteProject(project) {
  if (!confirm(`Delete project "${project.name}" permanently?`)) return;
  try {
    await api(`/api/projects/${project.id}`, { method: "DELETE" });
    if (state.currentProject === project.id) {
      state.currentProject = null;
      $("#chat").innerHTML = "";
      $("#file-list").innerHTML = "";
    }
    await loadProjects();
  } catch (e) {
    alert(e.message);
  }
}
async function clearProject(project) {
  if (!confirm(`Clear all messages in "${project.name}"?`)) return;
  try {
    await api(`/api/projects/${project.id}/messages`, { method: "DELETE" });
    await loadMessages();
    await loadUsage();
  } catch (e) {
    alert(e.message);
  }
}
async function moveProject(projectId, direction) {
  const ids = state.projects.map((p) => p.id);
  const idx = ids.indexOf(projectId);
  if (idx === -1) return;
  const target = idx + direction;
  if (target < 0 || target >= ids.length) return;
  [ids[idx], ids[target]] = [ids[target], ids[idx]];
  await reorderProjects(ids);
}
async function reorderProjects(ids) {
  try {
    await api("/api/projects/reorder", {
      method: "POST",
      body: JSON.stringify({ project_ids: ids }),
    });
    await loadProjects();
  } catch (e) {
    alert(e.message);
  }
}

/* ---------------- messages ---------------- */
async function loadMessages() {
  if (!state.currentProject) return;
  const d = await api(`/api/projects/${state.currentProject}/messages/v2`);
  state.messages = d.messages;
  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  state.lastUserMessage = lastUser ? lastUser.content : "";
  renderMessages();
}
function renderMessages() {
  const chat = $("#chat");
  chat.innerHTML = "";
  const lastAssistant = [...state.messages].reverse().find((m) => m.role === "assistant");
  for (const m of state.messages) {
    chat.appendChild(messageDom(m, lastAssistant && m.id === lastAssistant.id));
  }
  chat.scrollTop = chat.scrollHeight;
}
function messageDom(m, isLastAssistant) {
  const el = document.createElement("div");
  el.className = `msg ${m.role}`;
  const head = document.createElement("div");
  head.className = "msg-head";
  if (m.role === "assistant" && m.number) {
    const num = document.createElement("span");
    num.className = "msg-num";
    num.textContent = `#${m.number}`;
    head.appendChild(num);
  }
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy";
  copyBtn.onclick = () => navigator.clipboard.writeText(m.content);
  actions.appendChild(copyBtn);
  if (m.role === "user") {
    const editBtn = document.createElement("button");
    editBtn.textContent = "Edit";
    editBtn.onclick = () => editMessage(m);
    actions.appendChild(editBtn);
  }
  if (m.role === "assistant" && isLastAssistant) {
    const regenBtn = document.createElement("button");
    regenBtn.textContent = "Regenerate";
    regenBtn.onclick = () => streamChat(true);
    actions.appendChild(regenBtn);
  }
  const delBtn = document.createElement("button");
  delBtn.textContent = "Delete";
  delBtn.onclick = () => deleteMessage(m);
  actions.appendChild(delBtn);
  head.appendChild(actions);
  el.appendChild(head);
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = m.role === "assistant" ? md(m.content) : esc(m.content).replace(/\n/g, "<br>");
  el.appendChild(body);
  addMeta(el, m.meta || {});
  return el;
}
function addMeta(el, meta) {
  const chips = document.createElement("div");
  chips.className = "chip-row";
  for (const t of meta.router_tools || []) {
    chips.insertAdjacentHTML("beforeend", `<span class="chip router">⚡ ${esc(t)}</span>`);
  }
  for (const t of meta.tool_calls || []) {
    chips.insertAdjacentHTML("beforeend", `<span class="chip tool">🔧 ${esc(t.tool || "tool")}</span>`);
  }
  if ((meta.sources || []).length) {
    chips.insertAdjacentHTML("beforeend", `<span class="chip">📚 ${meta.sources.length} sources</span>`);
  }
  if (chips.childElementCount) el.appendChild(chips);
  for (const img of meta.images || []) {
    const i = document.createElement("img");
    i.src = img.url;
    i.alt = img.prompt || "generated image";
    i.onclick = () => window.open(img.url, "_blank");
    el.appendChild(i);
  }
  if ((meta.sources || []).length) {
    const d = document.createElement("details");
    d.className = "meta-box";
    d.innerHTML = "<summary>Sources</summary>";
    for (const s of meta.sources) {
      const item = document.createElement("div");
      item.className = "meta-item";
      item.textContent = `${s.metadata?.filename ? s.metadata.filename + " — " : ""}${s.text || ""}`;
      d.appendChild(item);
    }
    el.appendChild(d);
  }
}
function addTemporaryUser(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = esc(text).replace(/\n/g, "<br>");
  el.appendChild(body);
  $("#chat").appendChild(el);
  $("#chat").scrollTop = $("#chat").scrollHeight;
}
function addTemporaryAssistant() {
  const el = document.createElement("div");
  el.className = "msg assistant streaming";
  const head = document.createElement("div");
  head.className = "msg-head";
  const num = document.createElement("span");
  num.className = "msg-num";
  num.textContent = "…";
  head.appendChild(num);
  el.appendChild(head);
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = "";
  el.appendChild(body);
  $("#chat").appendChild(el);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return { el, body };
}
async function editMessage(m) {
  const val = prompt("Edit message:", m.content);
  if (val === null) return;
  try {
    await api(`/api/messages/${m.id}`, {
      method: "PATCH",
      body: JSON.stringify({ content: val }),
    });
    await loadMessages();
    await streamChat(true);
  } catch (e) {
    alert(e.message);
  }
}
async function deleteMessage(m) {
  if (!confirm("Delete this message?")) return;
  try {
    await api(`/api/messages/${m.id}`, { method: "DELETE" });
    await loadMessages();
    await loadUsage();
  } catch (e) {
    alert(e.message);
  }
}
async function streamChat(regenerate = false) {
  if (!state.currentProject) return alert("Select or create a project first.");
  if (state.streaming) return;
  const use_rag = $("#use-rag").checked;
  const agent_mode = $("#agent-mode").checked;
  let url;
  let body;
  if (regenerate) {
    if (!state.lastUserMessage) return alert("No user message to regenerate.");
    url = `/api/projects/${state.currentProject}/regenerate/stream`;
    body = { use_rag, agent_mode };
  } else {
    const text = $("#input").value.trim();
    if (!text) return;
    $("#input").value = "";
    addTemporaryUser(text);
    state.lastUserMessage = text;
    url = `/api/projects/${state.currentProject}/chat/stream`;
    body = {
      message: text,
      use_rag,
      agent_mode,
      file_ids: state.pendingFileIds,
    };
    state.pendingFileIds = [];
    renderPendingAttachments();
  }
  state.streaming = true;
  $("#send-btn").disabled = true;
  const node = addTemporaryAssistant();
  let acc = "";
  try {
    await ssePost(url, body, (ev, data) => {
      if (ev === "status") {
        node.body.textContent = `⏳ ${data.stage}…`;
      } else if (ev === "tool") {
        const tag = data.state === "start"
          ? `🔧 ${data.tool} → running…`
          : `✅ ${data.tool} done`;
        node.body.innerHTML = md((acc || "") + `\n\n\`${tag}\``);
      } else if (ev === "delta") {
        acc += data.text;
        node.body.textContent = acc;
        $("#chat").scrollTop = $("#chat").scrollHeight;
      } else if (ev === "done") {
        acc = data.answer || acc;
      } else if (ev === "error") {
        node.body.innerHTML = `<span class="status-error">Error: ${esc(data.message)}</span>`;
      }
    });
    await Promise.all([loadMessages(), loadUsage(), loadFiles(), loadModelSelect()]);
  } catch (e) {
    node.el.classList.remove("streaming");
    node.body.textContent = `Error: ${e.message}`;
  } finally {
    state.streaming = false;
    $("#send-btn").disabled = false;
  }
}

/* ---------------- files / attachments ---------------- */
async function loadFiles() {
  if (!state.currentProject) return;
  const d = await api(`/api/projects/${state.currentProject}/files`);
  state.files = d.files;
  state.pendingFileIds = state.pendingFileIds.filter((id) =>
    state.files.some((f) => f.id === id)
  );
  renderFiles();
}
function renderFiles() {
  const c = $("#file-list");
  c.innerHTML = "";
  if (!state.files.length) {
    c.innerHTML = '<div class="small">No files yet.</div>';
    renderPendingAttachments();
    return;
  }
  for (const f of state.files) {
    const item = document.createElement("div");
    item.className = "file-item" + (state.pendingFileIds.includes(f.id) ? " attached" : "");
    const line = document.createElement("div");
    line.className = "file-line";
    const a = document.createElement("a");
    a.href = `/api/files/${f.id}`;
    a.target = "_blank";
    a.textContent = `${f.kind === "image" ? "🖼️" : "📄"} ${f.filename}`;
    const actions = document.createElement("div");
    actions.className = "file-actions";
    const attachBtn = document.createElement("button");
    attachBtn.className = "tiny secondary";
    attachBtn.textContent = state.pendingFileIds.includes(f.id) ? "Detach" : "Attach";
    attachBtn.onclick = () => toggleAttach(f.id);
    const delBtn = document.createElement("button");
    delBtn.className = "tiny danger";
    delBtn.textContent = "✕";
    delBtn.title = "Remove file";
    delBtn.onclick = () => deleteFile(f.id);
    actions.append(attachBtn, delBtn);
    line.append(a, actions);
    item.appendChild(line);
    const meta = document.createElement("div");
    meta.className = "small";
    meta.textContent = `${f.kind} · ${(f.indexed_chars || 0).toLocaleString()} chars indexed`;
    item.appendChild(meta);
    if (f.kind === "image") {
      const img = document.createElement("img");
      img.src = `/api/files/${f.id}`;
      img.alt = f.filename;
      item.appendChild(img);
    }
    c.appendChild(item);
  }
  renderPendingAttachments();
}
function toggleAttach(fileId) {
  if (state.pendingFileIds.includes(fileId)) {
    state.pendingFileIds = state.pendingFileIds.filter((x) => x !== fileId);
  } else {
    state.pendingFileIds.push(fileId);
  }
  renderFiles();
}
function renderPendingAttachments() {
  const c = $("#pending-attachments");
  if (!state.pendingFileIds.length) {
    c.classList.add("hidden");
    c.innerHTML = "";
    return;
  }
  c.classList.remove("hidden");
  c.innerHTML = '<span class="small">Attachments:</span>';
  for (const id of state.pendingFileIds) {
    const f = state.files.find((x) => x.id === id);
    const chip = document.createElement("span");
    chip.className = "chip attachment";
    const label = document.createElement("span");
    label.textContent = f ? f.filename : String(id);
    const x = document.createElement("b");
    x.textContent = "✕";
    x.style.cursor = "pointer";
    x.onclick = () => toggleAttach(id);
    chip.append(label, x);
    c.appendChild(chip);
  }
}
async function uploadFiles(list) {
  if (!state.currentProject) return alert("Select a project first.");
  if (!list || !list.length) return;
  for (const f of list) {
    const fd = new FormData();
    fd.append("file", f);
    try {
      await api(`/api/projects/${state.currentProject}/upload`, {
        method: "POST",
        body: fd,
      });
    } catch (e) {
      alert(`${f.name}: ${e.message}`);
    }
  }
  await Promise.all([loadFiles(), loadUsage()]);
}
async function deleteFile(fileId) {
  if (!confirm("Remove this file from the project and RAG index?")) return;
  try {
    await api(`/api/files/${fileId}`, { method: "DELETE" });
    state.pendingFileIds = state.pendingFileIds.filter((x) => x !== fileId);
    await Promise.all([loadFiles(), loadUsage()]);
  } catch (e) {
    alert(e.message);
  }
}

/* ---------------- models ---------------- */
async function loadModelSelect() {
  try {
    const d = await api("/api/models/local");
    state.localModels = d.models;
    const sel = $("#model-select");
    const current = d.active.llm;
    let llms = d.models.filter((m) =>
      m.kind === "llm" || m.active_role === "llm" || m.id === current
    );
    if (!llms.some((m) => m.id === current)) {
      llms.unshift({ id: current, kind: "llm" });
    }
    sel.innerHTML = llms.map((m) =>
      `<option value="${esc(m.id)}" ${m.id === current ? "selected" : ""}>` +
      `${esc((m.id || "").split("/").pop())}${m.active_role === "llm" ? " *" : ""}` +
      `</option>`
    ).join("");
    sel.onchange = async () => {
      try {
        await api("/api/models/use", {
          method: "POST",
          body: JSON.stringify({ model_id: sel.value, kind: "llm" }),
        });
        await Promise.all([loadStatus(), loadUsage(), loadModelSelect()]);
      } catch (e) {
        alert(e.message);
      }
    };
  } catch (_) {}
}
const fmtBytes = (b) => !b ? "-" : b > 1024 ** 3 ? (b / 1024 ** 3).toFixed(2) + " GB" : (b / 1024 ** 2).toFixed(0) + " MB";
const fmtCtx = (n) => !n ? "-" : n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n);
async function loadModels() {
  try {
    if (state.localOnly) {
      const d = await api("/api/models/local");
      state.models = d.models;
      $("#disk-info").textContent = `disk free ${d.disk.free_gb} GB · models ${d.disk.models_gb} GB`;
    } else {
      const p = new URLSearchParams({
        search: $("#hub-search").value.trim(),
        task: $("#hub-task").value,
        sort: $("#hub-sort").value,
        limit: "40",
        deep: $("#hub-deep").checked ? "true" : "false",
      });
      const d = await api("/api/models/browse?" + p);
      state.models = d.models;
      $("#disk-info").textContent = `disk free ${d.disk.free_gb} GB · models ${d.disk.models_gb} GB`;
    }
    renderModels();
  } catch (e) {
    $("#hub-detail").classList.remove("hidden");
    $("#hub-detail").textContent = e.message;
  }
}
function renderModels() {
  const tb = $("#hub-table tbody");
  tb.innerHTML = "";
  for (const m of state.models) {
    const tr = document.createElement("tr");
    const caps = (m.capabilities || []).slice(0, 6).map((c) =>
      `<span class="badge ${["tools", "vision", "image-generation"].includes(c) ? "hot" : ""} ${c === "gated" ? "gated" : ""}">${esc(c)}</span>`
    ).join("");
    const installed = m.downloaded || m.status === "ready";
    
    // NEW: Determine Backend Badge
    const backendBadge = m.backend === "llama.cpp" 
      ? '<span class="badge hot">llama.cpp</span>' 
      : m.format === "gguf" 
        ? '<span class="badge hot">GGUF</span>'
        : '';

    const status = installed
      ? `<span class="status-ready">installed</span>`
      : m.status === "downloading"
        ? `<span class="status-downloading">${Math.round((m.progress || 0) * 100)}%</span>`
        : m.status === "error"
          ? `<span class="status-error" title="${esc(m.error || "")}">error</span>`
          : `<span class="small">not installed</span>`;
          
    tr.innerHTML = `
      <td class="mid">${esc(m.id)}
        ${m.active_role ? `<br><span class="badge active">${esc(m.active_role)}</span>` : ""}
        ${backendBadge ? `<br>${backendBadge}` : ""}
        ${m.note ? `<br><span class="small">${esc(m.note)}</span>` : ""}
      </td>
      <td>${esc(m.kind || "-")}</td>
      <td>${m.params_billion ? m.params_billion + "B" : "-"}</td>
      <td>${fmtBytes(m.local_bytes || m.size_bytes)}${m.size_estimated ? '<span class="small">~</span>' : ""}</td>
      <td>${fmtCtx(m.context_tokens)}</td>
      <td>${caps || "-"}</td>
      <td class="small">${m.downloads ? (m.downloads > 1e6 ? (m.downloads / 1e6).toFixed(1) + "M" : m.downloads > 1e3 ? (m.downloads / 1e3).toFixed(0) + "k" : m.downloads) : "-"}${m.likes ? ` · ♥${m.likes}` : ""}</td>
      <td>${status}${m.status === "downloading" ? `<div class="bar"><div class="bar-fill" style="width:${(m.progress || 0) * 100}%"></div></div>` : ""}</td>
      <td style="white-space:nowrap"></td>
    `;
    const td = tr.lastElementChild;
    if (!installed) {
      const dl = document.createElement("button");
      dl.className = "tiny";
      dl.textContent = "Download";
      dl.onclick = () => startDownload(m);
      td.appendChild(dl);
    } else {
      const use = document.createElement("button");
      use.className = "tiny";
      use.textContent = "Use";
      use.onclick = () => useModel(m);
      td.appendChild(use);
      const del = document.createElement("button");
      del.className = "tiny danger";
      del.textContent = "Del";
      del.onclick = () => delModel(m);
      td.appendChild(del);
    }
    const info = document.createElement("button");
    info.className = "tiny secondary";
    info.textContent = "i";
    info.onclick = () => showModelDetails(m);
    td.appendChild(info);
    tb.appendChild(tr);
  }
}
async function startDownload(m) {
  try {
    await api("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" }),
    });
    m.status = "downloading";
    m.progress = 0;
    renderModels();
    watchProgress();
  } catch (e) {
    alert(e.message);
  }
}
let progressWatcher = null;
function watchProgress() {
  if (progressWatcher) return;
  progressWatcher = setInterval(async () => {
    try {
      const d = await api("/api/models/jobs");
      const active = d.jobs.filter((j) => ["queued", "downloading"].includes(j.status));
      $("#hub-progress").innerHTML = active.length
        ? active.map((j) => `
            <div class="small">
              ${esc(j.repo_id)} — ${j.status} ${Math.round(j.progress * 100)}%
              (${fmtBytes(j.downloaded)}/${fmtBytes(j.total)}) ${esc(j.stage || "")}
              <div class="bar"><div class="bar-fill" style="width:${j.progress * 100}%"></div></div>
            </div>
          `).join("")
        : "";
      if (!active.length) {
        clearInterval(progressWatcher);
        progressWatcher = null;
        $("#hub-progress").innerHTML = "";
        await Promise.all([loadModels(), loadStatus(), loadModelSelect()]);
      }
    } catch (_) {
      clearInterval(progressWatcher);
      progressWatcher = null;
    }
  }, 1200);
}
async function useModel(m) {
  try {
    const r = await api("/api/models/use", {
      method: "POST",
      body: JSON.stringify({ model_id: m.id, kind: m.kind }),
    });
    alert(`Active ${r.field}: ${r.model}\nBackend: ${r.backend || 'transformers'}`);
    await Promise.all([loadModels(), loadStatus(), loadModelSelect(), loadUsage()]);
  } catch (e) {
    alert(e.message);
  }
}
async function delModel(m) {
  if (!confirm(`Delete ${m.id} from disk (${fmtBytes(m.local_bytes || m.size_bytes)})?`)) return;
  try {
    await api("/api/models/delete", {
      method: "POST",
      body: JSON.stringify({ model_id: m.id }),
    });
    await Promise.all([loadModels(), loadStatus(), loadModelSelect()]);
  } catch (e) {
    alert(e.message);
  }
}
async function showModelDetails(m) {
  const box = $("#hub-detail");
  box.classList.remove("hidden");
  box.textContent = "Loading details…";
  try {
    const d = await api("/api/models/details/" + encodeURIComponent(m.id));
    box.innerHTML = `
      <strong>${esc(d.id)}</strong> · ${esc(d.kind || "")} · ${esc(d.library || "")}
      ${d.backend ? `· <span class="badge hot">${esc(d.backend)}</span>` : ""}
      <div class="small">
        params ${d.params_billion ? d.params_billion + "B" : "-"} ·
        exact size ${fmtBytes(d.size_bytes)} ·
        context ${fmtCtx(d.context_tokens)} ·
        likes ${d.likes || 0} ·
        downloads ${d.downloads || 0}
        ${d.gated ? " · GATED" : ""}
        ${d.architecture ? " · " + esc(d.architecture) : ""}
      </div>
      <div>${(d.capabilities || []).map((c) => `<span class="badge">${esc(c)}</span>`).join("")}</div>
      ${d.error ? `<div class="status-error">${esc(d.error)}</div>` : ""}
      <div class="small">files: ${esc((d.files || []).slice(0, 25).join(", "))}</div>
    `;
  } catch (e) {
    box.textContent = e.message;
  }
}

/* ---------------- studio ---------------- */
async function generateImage() {
  const prompt = $("#img-prompt").value.trim();
  if (!prompt) return alert("Enter a prompt.");
  const body = {
    prompt,
    negative: $("#img-negative").value,
    steps: +$("#img-steps").value || 4,
    width: +$("#img-width").value || 512,
    height: +$("#img-height").value || 512,
    guidance: +$("#img-guidance").value || 0,
    project_id: state.currentProject,
  };
  const seed = +$("#img-seed").value;
  body.seed = seed > 0 ? seed : null;
  $("#img-go").disabled = true;
  $("#img-progress").classList.remove("hidden");
  try {
    await ssePost("/api/imagegen/stream", body, (ev, d) => {
      if (ev === "progress") {
        $("#img-progress .bar-fill").style.width = (d.progress * 100) + "%";
        $("#img-step").textContent = `${d.step || 0}/${d.total_steps || "?"} steps · ${Math.round(d.progress * 100)}%`;
      } else if (ev === "image") {
        $("#img-current").innerHTML = `<img src="${d.images[0].url}" alt=""><div class="small">${esc(d.images[0].prompt)} · seed ${d.images[0].seed} · ${d.images[0].width}×${d.images[0].height}</div>`;
      } else if (ev === "error") {
        alert(d.message);
      } else if (ev === "done") {
        $("#img-progress").classList.add("hidden");
        loadGallery();
      }
    });
  } catch (e) {
    alert(e.message);
    $("#img-progress").classList.add("hidden");
  } finally {
    $("#img-go").disabled = false;
  }
}
async function loadGallery() {
  try {
    const d = await api("/api/imagegen/gallery");
    const g = $("#img-gallery");
    g.innerHTML = "";
    for (const im of d.images) {
      const el = document.createElement("div");
      el.className = "gal-item";
      el.innerHTML = `
        <img src="${im.url}" loading="lazy">
        <div class="cap">${esc(im.prompt)}</div>
      `;
      const x = document.createElement("button");
      x.className = "x danger";
      x.textContent = "✕";
      x.onclick = async () => {
        if (!confirm("Delete image?")) return;
        await api(`/api/imagegen/${im.id}`, { method: "DELETE" });
        loadGallery();
      };
      el.appendChild(x);
      el.querySelector("img").onclick = () => window.open(im.url, "_blank");
      g.appendChild(el);
    }
    if (!d.images.length) {
      g.innerHTML = '<div class="small">Nothing generated yet.</div>';
    }
  } catch (_) {}
}

/* ---------------- settings ---------------- */
async function fillSettings() {
  try {
    const d = await api("/api/settings");
    const s = d.settings;
    $("#set-theme").value = state.theme;
    $("#set-model").value = s.model || "";
    $("#set-vision").value = s.vision_model || "";
    $("#set-embedding").value = s.embedding_model || "";
    $("#set-image").value = s.image_model || "";
    $("#set-tokens").value = s.max_new_tokens;
    $("#set-temp").value = s.temperature;
    $("#set-topp").value = s.top_p;
    $("#set-topk").value = s.top_k;
    $("#set-rep").value = s.repetition_penalty;
    $("#set-history").value = s.history_messages;
    $("#set-device").value = s.device;
    $("#set-dtype").value = s.dtype;
    $("#set-vismode").value = s.vision_mode;
    $("#set-rounds").value = s.max_tool_rounds;
    $("#set-router").checked = !!s.agent_router;
    $("#set-stream").checked = !!s.stream;
    $("#set-autoanalyze").checked = !!s.vision_auto_analyze;
    $("#set-chunk").value = s.chunk_size;
    $("#set-overlap").value = s.chunk_overlap;
    $("#set-topkrag").value = s.top_k_rag;
    $("#set-budget").value = s.context_budget;
    $("#img-steps").value = s.image_steps;
    $("#img-width").value = s.image_width;
    $("#img-height").value = s.image_height;
    $("#img-guidance").value = s.image_guidance;
    $("#img-negative").value = s.image_negative || "";
    $("#set-token").placeholder = d.hf_token_set
      ? "token saved (leave blank to keep)"
      : "hf_… (optional)";
  } catch (e) {
    console.error(e);
  }
}
async function saveSettings() {
  const token = $("#set-token").value.trim();
  const patch = {
    model: $("#set-model").value.trim(),
    vision_model: $("#set-vision").value.trim(),
    embedding_model: $("#set-embedding").value.trim(),
    image_model: $("#set-image").value.trim(),
    max_new_tokens: +$("#set-tokens").value,
    temperature: +$("#set-temp").value,
    top_p: +$("#set-topp").value,
    top_k: +$("#set-topk").value,
    repetition_penalty: +$("#set-rep").value,
    history_messages: +$("#set-history").value,
    device: $("#set-device").value,
    dtype: $("#set-dtype").value,
    vision_mode: $("#set-vismode").value,
    max_tool_rounds: +$("#set-rounds").value,
    agent_router: $("#set-router").checked,
    stream: $("#set-stream").checked,
    vision_auto_analyze: $("#set-autoanalyze").checked,
    chunk_size: +$("#set-chunk").value,
    chunk_overlap: +$("#set-overlap").value,
    top_k_rag: +$("#set-topkrag").value,
    context_budget: +$("#set-budget").value,
    image_steps: +$("#img-steps").value,
    image_width: +$("#img-width").value,
    image_height: +$("#img-height").value,
    image_guidance: +$("#img-guidance").value,
    image_negative: $("#img-negative").value,
  };
  if (token) patch.hf_token = token;
  $("#set-msg").textContent = "Saving…";
  try {
    await api("/api/settings", {
      method: "POST",
      body: JSON.stringify(patch),
    });
    $("#set-msg").textContent = "Saved ✓";
    $("#set-token").value = "";
    await Promise.all([loadStatus(), loadModelSelect(), loadUsage()]);
  } catch (e) {
    $("#set-msg").textContent = "Error: " + e.message;
  }
}

/* ---------------- status bar ---------------- */
function setBar(id, pct) {
  const fill = document.getElementById(id + "-fill");
  const text = document.getElementById(id + "-text");
  if (fill) fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
  if (text) text.textContent = Math.round(pct) + "%";
}
async function loadResources() {
  try {
    // Use raw fetch so a 404 doesn't throw an error and break the UI
    const res = await fetch("/api/system/resources");
    if (!res.ok) return;
    const r = await res.json();
    setBar("cpu", r.cpu_percent);
    $("#cpu-text").textContent = Math.round(r.cpu_percent) + "%";
    setBar("ram", r.ram_percent);
    $("#ram-text").textContent = `${r.ram_used_gb}/${r.ram_total_gb} GB`;
    if (r.gpu?.available) {
      setBar("gpu", r.gpu.memory_percent);
      $("#gpu-text").textContent =
        `${r.gpu.name || "GPU"} ${Math.round(r.gpu.used_mb / 1024)}/${Math.round(r.gpu.total_mb / 1024)} GB`;
    } else {
      setBar("gpu", 0);
      $("#gpu-text").textContent = "no GPU";
    }
    if (r.disk) {
      $("#disk-text").textContent = `${r.disk.free_gb}/${r.disk.total_gb} GB free`;
    }
  } catch (_) {}
}
async function loadUsage() {
  if (!state.user) return;
  try {
    const res = await fetch("/api/usage" + (state.currentProject ? `?project_id=${state.currentProject}` : ""));
    if (!res.ok) return;
    const d = await res.json();
    $("#stat-projects").textContent = `${d.projects.total} projects`;
    $("#stat-chats").textContent = `${d.chats.total_messages} msgs`;
    if (d.current) {
      $("#stat-tokens").textContent =
        `ctx ${d.current.tokens_used}/${d.current.context_limit} tok · ${d.current.tokens_available} free`;
    } else {
      $("#stat-tokens").textContent = "tokens: -";
    }
    $("#stat-model").textContent = `model: ${(d.model || "").split("/").pop() || "-"}`;
  } catch (_) {}
}
function startPolling() {
  setInterval(loadResources, 3000);
  setInterval(loadUsage, 10000);
  setInterval(loadStatus, 20000);
}

/* ---------------- voice ---------------- */
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    $("#voice-btn").disabled = true;
    $("#voice-btn").title = "Not supported in this browser";
    return;
  }
  state.recognition = new SR();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = navigator.language || "en-US";
  state.recognition.onresult = (e) => {
    $("#input").value = Array.from(e.results).map((r) => r[0].transcript).join("");
  };
  state.recognition.onend = () => {
    state.recognizing = false;
    $("#voice-btn").textContent = "🎤";
  };
  state.recognition.onerror = () => {
    state.recognizing = false;
    $("#voice-btn").textContent = "🎤";
  };
}
function toggleVoice() {
  if (!state.recognition) return alert("Speech recognition unsupported.");
  if (state.recognizing) {
    state.recognition.stop();
    state.recognizing = false;
    $("#voice-btn").textContent = "🎤";
  } else {
    state.recognition.start();
    state.recognizing = true;
    $("#voice-btn").textContent = "⏹";
  }
}

/* ---------------- bindings ---------------- */
function bind() {
  $("#login-btn").onclick = () => doAuth("/api/login");
  $("#register-btn").onclick = () => doAuth("/api/register");
  $("#new-project-btn").onclick = createProject;
  $("#send-btn").onclick = () => streamChat(false);
  $("#voice-btn").onclick = toggleVoice;
  $("#attach-btn").onclick = () => $("#file-input").click();
  $("#files-upload-btn").onclick = () => $("#file-input").click();
  $("#files-refresh-btn").onclick = loadFiles;
  $("#theme-btn").onclick = cycleTheme;
  $("#quick-settings-btn").onclick = () => switchTab("settings");
  $$(".tab").forEach((b) => b.onclick = () => switchTab(b.dataset.tab));
  $("#logout-btn").onclick = async () => {
    try { await api("/api/logout", { method: "POST" }); } catch (_) {}
    Object.assign(state, {
      user: null,
      projects: [],
      currentProject: null,
      files: [],
      messages: [],
      pendingFileIds: [],
    });
    $("#chat").innerHTML = "";
    $("#project-list").innerHTML = "";
    $("#file-list").innerHTML = "";
    renderAuthOnly();
  };
  $("#show-archived").onchange = (e) => {
    state.showArchived = e.target.checked;
    renderProjects();
  };
  $("#file-input").onchange = async (e) => {
    await uploadFiles(e.target.files);
    e.target.value = "";
  };
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      streamChat(false);
    }
  });
  $("#input").addEventListener("paste", async (e) => {
    const items = [...(e.clipboardData?.items || [])].filter((i) => i.type.startsWith("image/"));
    if (items.length) {
      e.preventDefault();
      for (const it of items) {
        const f = it.getAsFile();
        if (f) await uploadFiles([f]);
      }
    }
  });
  $("#project-rename").onclick = () => {
    const p = currentProject();
    if (p) renameProject(p);
  };
  $("#project-duplicate").onclick = () => {
    const p = currentProject();
    if (p) duplicateProject(p);
  };
  $("#project-export").onclick = () => {
    const p = currentProject();
    if (p) exportProject(p);
  };
  $("#project-pin").onclick = () => {
    const p = currentProject();
    if (p) patchProject(p.id, { pinned: !p.pinned });
  };
  $("#project-archive").onclick = () => {
    const p = currentProject();
    if (p) patchProject(p.id, { archived: !p.archived });
  };
  $("#project-move-up").onclick = () => {
    if (state.currentProject) moveProject(state.currentProject, -1);
  };
  $("#project-move-down").onclick = () => {
    if (state.currentProject) moveProject(state.currentProject, 1);
  };
  $("#project-clear").onclick = () => {
    const p = currentProject();
    if (p) clearProject(p);
  };
  $("#project-delete").onclick = () => {
    const p = currentProject();
    if (p) deleteProject(p);
  };
  $("#model-manage-btn").onclick = () => switchTab("models");
  $("#hub-refresh").onclick = () => {
    state.localOnly = false;
    $("#hub-local").textContent = "Installed only";
    loadModels();
  };
  $("#hub-local").onclick = () => {
    state.localOnly = !state.localOnly;
    $("#hub-local").textContent = state.localOnly ? "Browse Hub" : "Installed only";
    loadModels();
  };
  $("#hub-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      state.localOnly = false;
      loadModels();
    }
  });
  if ($("#hub-task")) {
    $("#hub-task").onchange = () => {
      state.localOnly = false;
      loadModels();
    };
  }
  if ($("#hub-sort")) {
    $("#hub-sort").onchange = () => loadModels();
  }
  $("#img-go").onclick = generateImage;
  $("#set-save").onclick = saveSettings;
  $("#set-theme").onchange = (e) => applyTheme(e.target.value);
  $("#set-reload").onclick = async () => {
    $("#set-msg").textContent = "Reloading engine…";
    try {
      await api("/api/engine/load", { method: "POST" });
      await loadStatus();
      $("#set-msg").textContent = "Engine reloaded ✓";
    } catch (e) {
      $("#set-msg").textContent = e.message;
    }
  };
  $("#unload-btn").onclick = async () => {
    if (!confirm("Unload all models from RAM/VRAM?")) return;
    await api("/api/engine/unload", { method: "POST" });
    await loadStatus();
  };
  document.addEventListener("click", closeMenu);
  $("#context-menu").addEventListener("click", (e) => e.stopPropagation());
}
boot();