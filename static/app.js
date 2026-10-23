const state = {
  user: null,
  projects: [],
  chats: [],
  currentChat: null,
  currentProject: null,
  expandedProject: null,
  files: [],
  messages: [],
  tab: "chat",
  models: [],
  localOnly: false,
  streaming: false,
  streamingChatId: null,
  streamAbortController: null,
  pendingFileIds: [],
  theme: localStorage.getItem("gpshap-theme") || "oceanic",
  showArchived: false,
  lastUserMessage: "",
  localModels: [],
  dismissedModelJobs: new Set(),
  polling: false,
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
async function openProjectResources(project) {
  const dialog = $("#resources-dialog");
  const list = $("#resources-list");
  $("#resources-title").textContent = `${project.name} resources`;
  list.replaceChildren();
  const loading = document.createElement("p");
  loading.className = "small";
  loading.textContent = "Loading resources…";
  list.appendChild(loading);
  $("#upload-resource-btn")?.remove();
  if (state.currentProject === project.id) {
    const upload = document.createElement("button");
    upload.id = "upload-resource-btn";
    upload.type = "button";
    upload.className = "secondary";
    upload.textContent = "Upload resources";
    upload.onclick = () => $("#file-input").click();
    $("#resources-dialog").querySelector(".dialog-header").appendChild(upload);
  }
  if (!dialog.open) dialog.showModal();

  try {
    const data = await api(`/api/projects/${project.id}/files`);
    list.replaceChildren();
    if (!data.files.length) {
      const empty = document.createElement("p");
      empty.className = "small";
      empty.textContent = "No resources in this project yet.";
      list.appendChild(empty);
      return;
    }
    for (const file of data.files) {
      const entry = document.createElement("div");
      entry.className = "resource-entry";
      const details = document.createElement("div");
      details.className = "resource-entry-name";
      const link = document.createElement("a");
      link.href = `/api/files/${file.id}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = file.filename;
      details.appendChild(link);
      const meta = document.createElement("div");
      meta.className = "small";
      meta.textContent = `${file.kind} · ${(file.indexed_chars || 0).toLocaleString()} indexed characters`;
      details.appendChild(meta);
      const actions = document.createElement("div");
      actions.className = "resource-entry-actions";
      if (state.currentProject === project.id) {
        const attach = document.createElement("button");
        attach.className = "tiny secondary";
        attach.type = "button";
        attach.textContent = state.pendingFileIds.includes(file.id) ? "Detach" : "Attach";
        attach.onclick = async () => {
          toggleAttach(file.id);
          await openProjectResources(project);
        };
        actions.appendChild(attach);
      }
      const remove = document.createElement("button");
      remove.className = "tiny danger";
      remove.type = "button";
      remove.textContent = "Remove";
      remove.onclick = async () => {
        if (!confirm(`Remove "${file.filename}" from this project?`)) return;
        try {
          await api(`/api/files/${file.id}`, { method: "DELETE" });
          if (state.currentProject === project.id) await loadFiles();
          await loadUsage();
          await openProjectResources(project);
        } catch (error) {
          alert(error.message);
        }
      };
      actions.appendChild(remove);
      entry.append(details, actions);
      list.appendChild(entry);
    }
  } catch (error) {
    list.replaceChildren();
    const failure = document.createElement("p");
    failure.className = "status-error";
    failure.textContent = `Could not load resources: ${error.message}`;
    list.appendChild(failure);
  }
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

async function ssePost(url, body, on, signal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
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
const THEMES = ["oceanic", "terminal", "zen"];
function applyTheme(theme) {
  if (!THEMES.includes(theme)) theme = "oceanic";
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  document.body.classList.remove(...THEMES.map((name) => `theme-${name}`));
  document.body.classList.add(`theme-${theme}`);
  if (theme !== "zen") document.body.classList.remove("sidebar-open");
  localStorage.setItem("gpshap-theme", theme);
  const sel = $("#set-theme");
  if (sel) sel.value = theme;
  const sidebarToggle = $("#sidebar-toggle");
  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded",
      String(document.body.classList.contains("sidebar-open")));
  }
}
function cycleTheme() {
  const idx = THEMES.indexOf(state.theme);
  applyTheme(THEMES[(idx + 1) % THEMES.length]);
}

/* ---------------- auth / boot ---------------- */
function renderAuthOnly() {
  $("#app").classList.add("hidden");
  $("#auth-screen").classList.remove("hidden");
}
function renderApp() {
  $("#auth-screen").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#project-panel").classList.remove("hidden");
  $("#username-label").textContent = state.user?.username || "User";
  $("#profile-name").textContent = state.user?.username || "";
  $("#profile-summary").setAttribute(
    "aria-label",
    `Profile for ${state.user?.username || "user"}`,
  );
}
async function boot() {
  const startedAt = performance.now();
  applyTheme(state.theme);
  bind();
  let authenticated = false;
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
    authenticated = true;
  } catch (_) {
    state.user = null;
  }
  const remaining = 3000 - (performance.now() - startedAt);
  if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
  $("#splash-screen").classList.add("hidden");
  if (authenticated) {
    renderApp();
    switchTab("chat");
    startPolling();
  } else {
    renderAuthOnly();
  }
}
async function doAuth(path) {
  const username = $("#auth-username").value.trim();
  const password = $("#auth-password").value;
  const error = $("#auth-error");
  error.textContent = "";
  error.classList.add("hidden");
  if (!username || !password) {
    error.textContent = "Enter username and password.";
    error.classList.remove("hidden");
    return;
  }
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
    error.textContent = e.message;
    error.classList.remove("hidden");
  }
}
async function loadStatus() {
  try {
    const s = await api("/api/status");
    const backend = s.llm.backend || "transformers";
    $("#studio-model").textContent = s.imagegen.model_id || "";
    $("#img-warn").textContent = s.imagegen.cpu_warning
      ? "⚠ CPU diffusion is very slow - lower steps or use a GPU."
      : "";
    $("#status-box").textContent =
      `LLM      : ${s.llm.model_id}\n` +
      `Backend  : ${backend}\n` +
      `device   : ${s.llm.device}${s.llm.dtype ? " · " + s.llm.dtype : ""}\n` +
      `params   : ${s.llm.params_billion ? s.llm.params_billion + "B" : "-"}\n` +
      `context  : ${s.llm.context_tokens || "-"} tok\n` +
      (s.llm.vram_allocated_mb ? `vram     : ${s.llm.vram_allocated_mb} MB\n` : "") +
      `vision   : ${s.vision.model_id} ${s.vision.downloaded ? "(ready)" : "(not downloaded)"}\n` +
      `embed    : ${s.embedding.model}\n` +
      `image    : ${s.imagegen.model_id}\n` +
      `disk free: ${s.disk.free_gb ?? "-"} GB (models ${s.disk.models_gb ?? "-"} GB)`;
    $("#runtime-engine").textContent =
      `Engine: ${s.llm.loaded ? "LLM ready" : "loading"} · ${s.llm.device}` +
      `${s.llm.cuda_available ? " · " + (s.llm.gpu_name || "CUDA") : ""} · ${backend}`;
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
function isContextMenuTrigger(target) {
  return target instanceof Element &&
    target.closest(".p-menu, .chat-item-menu, #project-more-btn");
}
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
function dismissMenuOnOutsideClick(event) {
  const menu = $("#context-menu");
  if (menu.classList.contains("hidden")) return;
  if (menu.contains(event.target) || isContextMenuTrigger(event.target)) return;
  closeMenu();
}

/* ---------------- projects ---------------- */
async function loadProjects() {
  const d = await api("/api/projects/v2");
  state.projects = d.projects;
  const chats = await api("/api/chats");
  state.chats = chats.chats;
  renderProjects();
  if (!state.chats.some((chat) => chat.id === state.currentChat)) {
    if (state.chats.length) {
      await selectChat(state.chats[0].id);
    } else {
      await createChat();
    }
  }
  await loadUsage();
}
async function loadChats() {
  const d = await api("/api/chats");
  state.chats = d.chats;
  renderProjects();
}
function currentProject() {
  return state.projects.find((p) => p.id === state.currentProject) || null;
}
function activeChat() {
  return state.chats.find((chat) => chat.id === state.currentChat) || null;
}
function renderProjects() {
  const c = $("#project-list");
  c.innerHTML = "";
  const list = state.projects.filter((p) => state.showArchived || !p.archived);
  const generalChats = state.chats.filter((chat) => chat.project_id == null);
  const general = document.createElement("section");
  general.className = "chat-group";
  const generalTitle = document.createElement("div");
  generalTitle.className = "chat-group-title";
  generalTitle.textContent = "General chats";
  general.appendChild(generalTitle);
  if (!generalChats.length) {
    const empty = document.createElement("div");
    empty.className = "small chat-list-empty";
    empty.textContent = "No general chats";
    general.appendChild(empty);
  }
  for (const chat of generalChats) general.appendChild(chatItem(chat));
  c.appendChild(general);

  if (!list.length) {
    const empty = document.createElement("div");
    empty.className = "small chat-list-empty";
    empty.textContent = "No projects yet.";
    c.appendChild(empty);
  }
  for (const p of list) {
    const group = document.createElement("section");
    group.className = "chat-group";
    const el = document.createElement("div");
    el.className = "project-item" + (state.currentProject === p.id ? " active" : "") +
      (p.archived ? " archived" : "");
    el.draggable = true;
    el.dataset.id = String(p.id);
    const name = document.createElement("button");
    name.className = "project-name-button";
    name.type = "button";
    name.setAttribute("aria-expanded", String(state.expandedProject === p.id));
    name.textContent = (p.pinned ? "📌 " : "") + p.name;
    name.onclick = () => selectProject(p.id);
    const actions = document.createElement("div");
    actions.className = "project-item-actions";
    const resourcesBtn = document.createElement("button");
    resourcesBtn.className = "tiny secondary";
    resourcesBtn.type = "button";
    resourcesBtn.textContent = "Resources";
    resourcesBtn.title = `View resources for ${p.name}`;
    resourcesBtn.onclick = (e) => {
      e.stopPropagation();
      openProjectResources(p);
    };
    const menuBtn = document.createElement("button");
    menuBtn.className = "tiny secondary p-menu";
    menuBtn.type = "button";
    menuBtn.title = "Project options";
    menuBtn.textContent = "⋮";
    menuBtn.onclick = (e) => {
      e.stopPropagation();
      openProjectMenu(e.clientX, e.clientY, p);
    };
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
    actions.append(resourcesBtn, menuBtn);
    el.append(name, actions);
    group.appendChild(el);
    if (state.expandedProject === p.id) {
      const projectChats = state.chats.filter((chat) => chat.project_id === p.id);
      for (const chat of projectChats) group.appendChild(chatItem(chat));
      const addChat = document.createElement("button");
      addChat.className = "project-add-chat";
      addChat.type = "button";
      addChat.textContent = "+ New chat";
      addChat.onclick = () => createChat(p.id);
      group.appendChild(addChat);
    }
    c.appendChild(group);
  }
}
function chatItem(chat) {
  const item = document.createElement("div");
  item.className = "chat-item" + (state.currentChat === chat.id ? " active" : "");
  item.title = chat.title;
  const title = document.createElement("span");
  title.className = "chat-item-title";
  title.textContent = chat.title || "New chat";
  const menu = document.createElement("button");
  menu.className = "chat-item-menu";
  menu.type = "button";
  menu.setAttribute("aria-label", `Options for ${title.textContent}`);
  menu.textContent = "⋮";
  menu.onclick = (event) => {
    event.stopPropagation();
    openChatMenu(event.clientX, event.clientY, chat);
  };
  item.onclick = () => selectChat(chat.id);
  item.append(title, menu);
  return item;
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
function openChatMenu(x, y, chat) {
  openMenu(x, y, [
    { label: "Rename chat", onClick: () => renameChat(chat) },
    { label: "Duplicate chat", onClick: () => duplicateChat(chat) },
    { label: "Export chat", onClick: () => exportChat(chat) },
    { label: "Clear chat", onClick: () => clearChat(chat) },
    { sep: true },
    { label: "Delete chat", danger: true, onClick: () => deleteChat(chat) },
  ]);
}
async function createProject() {
  const name = prompt("Project name:");
  if (!name || !name.trim()) return;
  const sys = prompt("Optional system prompt:", "") || "";
  try {
    const created = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: name.trim(), system_prompt: sys }),
    });
    await loadProjects();
    await createChat(created.id);
  } catch (e) {
    alert(e.message);
  }
}
async function selectProject(id) {
  state.expandedProject = id;
  renderProjects();
  const existingChat = state.chats.find((chat) => chat.project_id === id);
  if (existingChat) return selectChat(existingChat.id);
  return createChat(id);
}
async function createChat(projectId = null) {
  try {
    const chat = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({ project_id: projectId }),
    });
    await loadChats();
    await selectChat(chat.id);
  } catch (error) {
    alert(error.message);
  }
}
async function selectChat(id) {
  state.currentChat = id;
  const chat = state.chats.find((entry) => entry.id === id);
  state.currentProject = chat?.project_id ?? null;
  state.expandedProject = state.currentProject;
  state.pendingFileIds = [];
  renderProjects();
  $("#current-project").textContent = chat?.title || "New chat";
  $("#files-upload-btn").disabled = !state.currentProject;
  if (state.currentProject) {
    await Promise.all([loadMessages(), loadFiles(), loadUsage()]);
  } else {
    state.files = [];
    $("#file-list").replaceChildren();
    renderPendingAttachments();
    await Promise.all([loadMessages(), loadUsage()]);
  }
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
async function renameChat(chat) {
  const title = prompt("Rename chat:", chat.title);
  if (!title || !title.trim()) return;
  try {
    await api(`/api/chats/${chat.id}`, {
      method: "PATCH",
      body: JSON.stringify({ title: title.trim() }),
    });
    await loadChats();
    if (state.currentChat === chat.id) {
      $("#current-project").textContent = title.trim();
    }
  } catch (error) {
    alert(error.message);
  }
}
async function duplicateChat(chat) {
  try {
    const result = await api(`/api/chats/${chat.id}/duplicate`, { method: "POST" });
    await loadChats();
    await selectChat(result.id);
  } catch (error) {
    alert(error.message);
  }
}
async function exportChat(chat) {
  try {
    const data = await api(`/api/chats/${chat.id}/export`);
    downloadJSON(`gpshap_chat_${chat.id}.json`, data);
  } catch (error) {
    alert(error.message);
  }
}
async function clearChat(chat) {
  if (!confirm(`Clear all messages in "${chat.title}"?`)) return;
  try {
    await api(`/api/chats/${chat.id}/messages`, { method: "DELETE" });
    if (state.currentChat === chat.id) await loadMessages();
    await loadChats();
    await loadUsage();
  } catch (error) {
    alert(error.message);
  }
}
async function deleteChat(chat) {
  if (!confirm(`Delete chat "${chat.title}" permanently?`)) return;
  try {
    await api(`/api/chats/${chat.id}`, { method: "DELETE" });
    if (state.currentChat === chat.id) {
      state.currentChat = null;
      state.currentProject = null;
      state.messages = [];
      renderMessages();
    }
    await loadProjects();
  } catch (error) {
    alert(error.message);
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
  if (!state.currentChat) return;
  const d = await api(`/api/chats/${state.currentChat}/messages/v2`);
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
  $("#chat-empty-actions").classList.toggle(
    "hidden",
    !state.currentChat || Boolean(lastAssistant),
  );
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
  el.appendChild(head);
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = m.role === "assistant" ? md(m.content) : esc(m.content).replace(/\n/g, "<br>");
  el.appendChild(body);
  addMeta(el, m.meta || {});
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copyBtn = document.createElement("button");
  copyBtn.className = "message-action";
  copyBtn.type = "button";
  copyBtn.setAttribute("aria-label", m.role === "assistant" ? "Copy response" : "Copy message");
  copyBtn.title = "Copy";
  copyBtn.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>';
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(m.content);
      copyBtn.title = "Copied";
    } catch (error) {
      alert(`Could not copy message: ${error.message}`);
    }
  };
  actions.appendChild(copyBtn);
  if (m.role === "assistant") {
    for (const [label, icon] of [
      ["Like response", "&#128077;"],
      ["Dislike response", "&#128078;"],
    ]) {
      const vote = document.createElement("button");
      vote.className = "message-action vote-action";
      vote.type = "button";
      vote.setAttribute("aria-label", label);
      vote.setAttribute("aria-pressed", "false");
      vote.title = label;
      vote.innerHTML = icon;
      vote.onclick = () => {
        const wasPressed = vote.getAttribute("aria-pressed") === "true";
        actions.querySelectorAll(".vote-action").forEach((button) => {
          button.setAttribute("aria-pressed", "false");
        });
        vote.setAttribute("aria-pressed", String(!wasPressed));
      };
      actions.appendChild(vote);
    }
  }
  const moreBtn = document.createElement("button");
  moreBtn.className = "message-action";
  moreBtn.type = "button";
  moreBtn.setAttribute("aria-label", "More chat actions");
  moreBtn.title = "More";
  moreBtn.textContent = "⋯";
  moreBtn.onclick = (event) => {
    event.stopPropagation();
    openMessageMenu(event, m, isLastAssistant);
  };
  actions.appendChild(moreBtn);
  el.appendChild(actions);
  return el;
}
function openMessageMenu(event, message, isLastAssistant) {
  event.stopPropagation();
  const chat = activeChat();
  const items = [];
  if (message.role === "user") {
    items.push({ label: "Edit message", onClick: () => editMessage(message) });
  }
  items.push({ label: "Delete message", danger: true, onClick: () => deleteMessage(message) });
  if (message.role === "assistant" && isLastAssistant) {
    items.unshift({ label: "Regenerate response", onClick: () => streamChat(true) });
  }
  if (isLastAssistant && chat) {
    items.push(
      { sep: true },
      { label: "Rename chat", onClick: () => renameChat(chat) },
      { label: "Duplicate chat", onClick: () => duplicateChat(chat) },
      { label: "Export chat", onClick: () => exportChat(chat) },
      { label: "Clear chat", onClick: () => clearChat(chat) },
      { label: "Delete chat", danger: true, onClick: () => deleteChat(chat) },
    );
  }
  openMenu(event.clientX, event.clientY, items);
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
  if ((meta.web_sources || []).length) {
    chips.insertAdjacentHTML("beforeend", `<span class="chip">🌐 ${meta.web_sources.length} web results</span>`);
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
  if ((meta.web_sources || []).length || meta.web_search_error) {
    const d = document.createElement("details");
    d.className = "meta-box";
    const summary = document.createElement("summary");
    summary.textContent = "Web search";
    d.appendChild(summary);
    if (meta.web_search_error) {
      const error = document.createElement("div");
      error.className = "meta-item status-error";
      error.textContent = `Search failed: ${meta.web_search_error}`;
      d.appendChild(error);
    }
    for (const source of meta.web_sources || []) {
      const item = document.createElement("div");
      item.className = "meta-item";
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      item.appendChild(link);
      if (source.snippet) {
        const snippet = document.createElement("div");
        snippet.textContent = source.snippet;
        item.appendChild(snippet);
      }
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
  if (!state.currentChat) return alert("Create or select a chat first.");
  if (state.streaming) return;
  const use_rag = $("#use-rag").checked;
  const agent_mode = $("#agent-mode").checked;
  const use_web = $("#use-web").checked;
  let url;
  let body;
  if (regenerate) {
    if (!state.lastUserMessage) return alert("No user message to regenerate.");
    url = `/api/chats/${state.currentChat}/regenerate/stream`;
    body = { use_rag, agent_mode, use_web };
  } else {
    const text = $("#input").value.trim();
    if (!text) return;
    $("#input").value = "";
    addTemporaryUser(text);
    state.lastUserMessage = text;
    url = `/api/chats/${state.currentChat}/chat/stream`;
    body = {
      message: text,
      use_rag,
      agent_mode,
      use_web,
      file_ids: state.pendingFileIds,
    };
    state.pendingFileIds = [];
    renderPendingAttachments();
  }
  state.streaming = true;
  state.streamingChatId = state.currentChat;
  const controller = new AbortController();
  state.streamAbortController = controller;
  $("#send-btn").classList.add("hidden");
  $("#stop-btn").classList.remove("hidden");
  
  const node = addTemporaryAssistant();
  let acc = "";
  let streamError = "";
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
        streamError = data.message || "The model could not complete this response.";
        node.el.classList.remove("streaming");
        node.body.innerHTML = `<span class="status-error">Error: ${esc(streamError)}</span>`;
      }
    }, controller.signal);
    await Promise.all([loadUsage(), loadChats(), loadFiles(), loadModelSelect()]);
    const chat = activeChat();
    if (chat) $("#current-project").textContent = chat.title;
    if (!streamError) await loadMessages();
  } catch (e) {
    node.el.classList.remove("streaming");
    if (e.name === "AbortError") {
      node.body.textContent = `${acc}${acc ? "\n\n" : ""}[Stopped]`;
    } else {
      node.body.textContent = `Error: ${e.message}`;
    }
  } finally {
    state.streaming = false;
    state.streamingChatId = null;
    state.streamAbortController = null;
    $("#send-btn").classList.remove("hidden");
    $("#stop-btn").classList.add("hidden");
  }
}

async function stopChat() {
  if (!state.streaming) return;
  if (!state.streamingChatId) return;
  try {
    await api(`/api/chats/${state.streamingChatId}/chat/stop`, { method: "POST" });
  } catch (e) {
    console.error("Failed to stop chat", e);
    alert(`The server could not confirm cancellation. Generation may continue in the background: ${e.message}`);
  } finally {
    state.streamAbortController?.abort();
  }
}

/* ---------------- files / attachments ---------------- */
async function loadFiles() {
  if (!state.currentProject) {
    state.files = [];
    renderFiles();
    return;
  }
  const d = await api(`/api/projects/${state.currentProject}/files`);
  state.files = d.files;
  state.pendingFileIds = state.pendingFileIds.filter((id) =>
    state.files.some((f) => f.id === id)
  );
  renderFiles();
}
function renderFiles() {
  const c = $("#file-list");
  if (!c) return;
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
  const project = currentProject();
  if (project && $("#resources-dialog").open) await openProjectResources(project);
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
      `${esc((m.id || "").split("/").pop())}` +
      `${m.format === "gguf" ? " [GGUF - llama.cpp]" : ""}` +
      `${m.active_role === "llm" ? " *" : ""}` +
      `</option>`
    ).join("");
    sel.onchange = async () => {
      try {
        const selected = llms.find((m) => m.id === sel.value);
        const body = { model_id: sel.value, kind: "llm" };
        if (selected?.format === "gguf" || sel.value.toLowerCase().endsWith(".gguf")) {
          const inventory = await api("/api/models/local");
          Object.assign(body, inventory.gguf_config || {});
        }
        await api("/api/models/use", {
          method: "POST",
          body: JSON.stringify(body),
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
    state.models = state.models.filter(
      (model) => !state.dismissedModelJobs.has(model.id)
    );
    const jobData = await api("/api/models/jobs");
    const jobsById = new Map((jobData.jobs || []).map((job) => [job.repo_id, job]));
    for (const model of state.models) {
      const job = jobsById.get(model.id);
      if (job && !state.dismissedModelJobs.has(model.id)) {
        model.status = job.status;
        model.progress = job.progress;
        model.error = job.error;
      }
    }
    for (const job of jobData.jobs || []) {
      if (["queued", "downloading", "cancelling"].includes(job.status) &&
          !state.models.some((model) => model.id === job.repo_id)) {
        state.models.push({
          id: job.repo_id,
          kind: job.kind || "llm",
          status: job.status,
          progress: job.progress,
          downloaded: false,
        });
      }
    }
    renderModels();
    if ((jobData.jobs || []).some((job) =>
      ["queued", "downloading", "cancelling"].includes(job.status)
    )) {
      watchProgress();
    }
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
    const running = ["queued", "downloading"].includes(m.status);
    const installed = !running && (m.downloaded || m.status === "ready");
    
    // NEW: Determine Backend Badge
    const backendBadge = m.backend === "llama.cpp" 
      ? '<span class="badge hot">llama.cpp</span>' 
      : m.format === "gguf" 
        ? '<span class="badge hot">GGUF</span>'
        : '';

    let status;
    if (running) {
      status = `<span class="status-downloading">${Math.round((m.progress || 0) * 100)}%</span>`;
    } else if (m.status === "cancelling") {
      status = '<span class="status-downloading">cancellation requested</span>';
    } else if (m.status === "cancelled") {
      status = `<span class="small">${m.paused ? "cancelled; ready to resume" : "cancelled"}</span>`;
    } else if (m.status === "error") {
      status = `<span class="status-error" title="${esc(m.error || "")}">error</span>`;
    } else if (installed) {
      status = '<span class="status-ready">installed</span>';
    } else {
      status = '<span class="small">not installed</span>';
    }
          
    tr.innerHTML = `
      <td class="mid">${esc(m.id)}
        ${m.active_role ? `<br><span class="badge active">${esc(m.active_role)}</span>` : ""}
        ${backendBadge ? `<br>${backendBadge}` : ""}
        ${m.note ? `<br><span class="small">${esc(m.note)}</span>` : ""}
      </td>
      <td>${esc(m.kind || "-")}</td>
      <td>${m.params_billion ? m.params_billion + "B" : "-"}</td>
      <td title="${m.local_bytes != null ? "Actual files currently on disk" : m.size_estimated ? "Estimated from model metadata" : "Reported model repository size"}">
        ${fmtBytes(m.local_bytes ?? m.size_bytes)}
        ${m.local_bytes != null ? '<span class="small">(on disk)</span>' :
          m.size_estimated ? '<span class="small">(estimate)</span>' : ""}
      </td>
      <td>${fmtCtx(m.context_tokens)}</td>
      <td>${caps || "-"}</td>
      <td class="small">${m.downloads ? (m.downloads > 1e6 ? (m.downloads / 1e6).toFixed(1) + "M" : m.downloads > 1e3 ? (m.downloads / 1e3).toFixed(0) + "k" : m.downloads) : "-"}${m.likes ? ` · ♥${m.likes}` : ""}</td>
      <td>${status}${running || m.status === "cancelling" ? `<div class="bar"><div class="bar-fill" style="width:${(m.progress || 0) * 100}%"></div></div>` : ""}</td>
      <td style="white-space:nowrap"></td>
    `;
    const td = tr.lastElementChild;
    if (running) {
      const pauseBtn = document.createElement("button");
      pauseBtn.className = "tiny secondary";
      pauseBtn.textContent = "Pause (request cancel)";
      pauseBtn.title = "Pausing is unsupported. Cancellation is requested; the current download operation may need to finish first.";
      pauseBtn.onclick = () => pauseDownload(m);
      td.appendChild(pauseBtn);

      const cancelBtn = document.createElement("button");
      cancelBtn.className = "tiny danger";
      cancelBtn.textContent = "Cancel job";
      cancelBtn.onclick = () => cancelDownload(m, true);
      td.appendChild(cancelBtn);
    } else if (m.status === "cancelling") {
      const cancelling = document.createElement("span");
      cancelling.className = "status-downloading";
      cancelling.textContent = "Cancelling…";
      cancelling.title = "The current download operation must finish before resume is available.";
      td.appendChild(cancelling);
    } else if (m.status === "cancelled") {
      const resumeBtn = document.createElement("button");
      resumeBtn.className = "tiny";
      resumeBtn.textContent = "Resume";
      resumeBtn.title = "Restart this download; partial files are reused when supported.";
      resumeBtn.onclick = () => startDownload(m);
      td.appendChild(resumeBtn);
      const deleteJobBtn = document.createElement("button");
      deleteJobBtn.className = "tiny danger";
      deleteJobBtn.textContent = "Delete job";
      deleteJobBtn.onclick = () => dismissModelJob(m);
      td.appendChild(deleteJobBtn);
    } else if (!installed) {
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
    const result = await api("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" }),
    });
    if (result.ok === false) throw new Error(result.error || "Download could not be started.");
    m.status = "downloading";
    m.progress = 0;
    m.paused = false;
    state.dismissedModelJobs.delete(m.id);
    renderModels();
    watchProgress();
  } catch (e) {
    alert(e.message);
  }
}

async function pauseDownload(m) {
  await cancelDownload(m, false);
  if (m.status === "cancelling") m.paused = true;
  renderModels();
}
async function cancelDownload(m, dismiss = false) {
  try {
    await api("/api/models/cancel", {
      method: "POST",
      body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" }),
    });
    m.status = "cancelling";
    m.paused = false;
    if (dismiss) {
      await api("/api/models/jobs/delete", {
        method: "POST",
        body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" }),
      });
      state.dismissedModelJobs.add(m.id);
      state.models = state.models.filter((model) => model.id !== m.id);
    }
    renderModels();
  } catch (e) {
    renderModels();
    alert(e.message);
  }
}
function dismissModelJob(m) {
  api("/api/models/jobs/delete", {
    method: "POST",
    body: JSON.stringify({ model_id: m.id, kind: m.kind || "llm" }),
  }).then(() => {
    state.dismissedModelJobs.add(m.id);
    state.models = state.models.filter((model) => model.id !== m.id);
    renderModels();
  }).catch((e) => alert(e.message));
}

let progressWatcher = null;
function watchProgress() {
  if (progressWatcher) return;
  progressWatcher = setInterval(async () => {
    try {
      const d = await api("/api/models/jobs");
      const active = d.jobs.filter((j) =>
        ["queued", "downloading", "cancelling"].includes(j.status)
      );
      for (const job of d.jobs) {
        const model = state.models.find((entry) => entry.id === job.repo_id);
        if (model && !state.dismissedModelJobs.has(job.repo_id)) {
          model.status = job.status;
          model.progress = job.progress;
          model.error = job.error;
        }
      }
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
    await loadModels();
  } catch (e) {
    alert(e.message);
  }
}
function showModelDetails(m) {
  const d = $("#hub-detail");
  d.classList.remove("hidden");
  d.textContent = JSON.stringify(m, null, 2);
}

async function loadGallery() {
  try {
    const data = await api("/api/imagegen/gallery");
    const gallery = $("#img-gallery");
    gallery.replaceChildren();
    for (const image of data.images) {
      const item = document.createElement("div");
      item.className = "gal-item";
      const img = document.createElement("img");
      img.src = image.url;
      img.alt = image.prompt || "Generated image";
      img.loading = "lazy";
      img.onclick = () => window.open(image.url, "_blank", "noopener");
      item.appendChild(img);
      const caption = document.createElement("div");
      caption.className = "cap";
      caption.textContent = image.prompt || image.filename;
      item.appendChild(caption);
      gallery.appendChild(item);
    }
  } catch (error) {
    $("#img-message").textContent = `Could not load image gallery: ${error.message}`;
  }
}

async function generateImage() {
  const prompt = $("#img-prompt").value.trim();
  if (!prompt) {
    $("#img-message").textContent = "Enter an image prompt first.";
    $("#img-prompt").focus();
    return;
  }

  const button = $("#img-go");
  const progress = $("#img-progress");
  const fill = progress.querySelector(".bar-fill");
  const step = $("#img-step");
  const status = $("#img-message");
  let streamError = "";
  button.disabled = true;
  progress.classList.remove("hidden");
  fill.style.width = "0%";
  status.textContent = "Preparing image generation…";
  $("#img-current").replaceChildren();

  try {
    await ssePost("/api/imagegen/stream", {
      prompt,
      negative: $("#img-negative").value,
      steps: Number($("#img-steps").value),
      guidance: Number($("#img-guidance").value),
      width: Number($("#img-width").value),
      height: Number($("#img-height").value),
      seed: Number($("#img-seed").value) < 0 ? null : Number($("#img-seed").value),
      project_id: state.currentProject,
      chat_id: state.currentChat,
    }, (event, data) => {
      if (event === "status") {
        status.textContent = data.stage || "Preparing image model…";
        if (data.stage?.startsWith("First-time setup")) {
          step.textContent = "First-time model setup may take several minutes.";
        }
      } else if (event === "progress") {
        if (data.stage === "generation") {
          fill.style.width = `${Math.round((data.progress || 0) * 100)}%`;
          step.textContent = data.total_steps
            ? `Generating image: step ${data.step}/${data.total_steps}`
            : "Loading image-generation components…";
        } else {
          const downloaded = fmtBytes(data.downloaded);
          const total = data.total ? ` of ${fmtBytes(data.total)}` : "";
          step.textContent = `${data.message || "Downloading image model"} ${downloaded}${total}`;
        }
      } else if (event === "image") {
        const result = data.images?.[0];
        if (result) {
          const img = document.createElement("img");
          img.src = result.url;
          img.alt = result.prompt || "Generated image";
          $("#img-current").replaceChildren(img);
          fill.style.width = "100%";
          status.textContent = "Image generated and saved locally.";
        }
      } else if (event === "error") {
        streamError = data.message || "Image generation failed.";
        status.textContent = streamError;
      } else if (event === "done" && !streamError) {
        status.textContent = "Image generation complete.";
      }
    });
    if (streamError) throw new Error(streamError);
    await loadGallery();
  } catch (error) {
    status.textContent = `Image generation failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

/* ---------------- bindings ---------------- */
function bind() {
  document.addEventListener("click", dismissMenuOnOutsideClick);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
  });
  $("#auth-form").onsubmit = (event) => {
    event.preventDefault();
    doAuth("/api/login");
  };
  $("#register-btn").onclick = () => doAuth("/api/register");
  $("#password-form").onsubmit = async (event) => {
    event.preventDefault();
    const error = $("#password-error");
    error.classList.add("hidden");
    error.textContent = "";
    const currentPassword = $("#current-password").value;
    const newPassword = $("#new-password").value;
    if (newPassword !== $("#confirm-password").value) {
      error.textContent = "New passwords do not match.";
      error.classList.remove("hidden");
      return;
    }
    try {
      await api("/api/password", {
        method: "POST",
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });
      $("#password-dialog").close();
      $("#password-form").reset();
      alert("Password updated. Other active sessions have been signed out.");
    } catch (e) {
      error.textContent = e.message;
      error.classList.remove("hidden");
    }
  };
  $("#change-password-btn").onclick = () => {
    $("#user-menu").open = false;
    $("#password-dialog").showModal();
    $("#current-password").focus();
  };
  $("#password-close").onclick = () => $("#password-dialog").close();
  $("#feedback-btn").onclick = () => {
    $("#user-menu").open = false;
    $("#feedback-status").textContent = "";
    $("#feedback-dialog").showModal();
    $("#feedback-message").focus();
  };
  $("#feedback-close").onclick = () => $("#feedback-dialog").close();
  $("#feedback-form").onsubmit = (event) => {
    event.preventDefault();
    const message = $("#feedback-message").value.trim();
    if (!message) return;
    const subject = encodeURIComponent("Gpshap feedback");
    const body = encodeURIComponent(message);
    window.location.href = `mailto:gpshap@gmail.com?subject=${subject}&body=${body}`;
    $("#feedback-status").textContent =
      "Your email app should open with this draft. Review it and send it there; Gpshap does not send or save feedback.";
  };
  $("#resources-close").onclick = () => $("#resources-dialog").close();
  $("#logout-btn").onclick = async () => {
    await api("/api/logout", { method: "POST" });
    state.user = null;
    $("#user-menu").open = false;
    renderAuthOnly();
  };
  $("#new-chat-btn").onclick = () => createChat();
  $("#new-project-btn").onclick = createProject;
  $("#project-more-btn").onclick = (event) => {
    event.stopPropagation();
    const project = currentProject();
    if (project) openProjectMenu(event.clientX, event.clientY, project);
  };
  $("#show-archived").onchange = (e) => {
    state.showArchived = e.target.checked;
    renderProjects();
  };
  $("#files-upload-btn").onclick = () => $("#file-input").click();
  $("#file-input").onchange = (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";
  };
  $$(".tab").forEach((b) => {
    b.onclick = () => switchTab(b.dataset.tab);
  });
  $("#sidebar-toggle").onclick = () => {
    document.body.classList.toggle("sidebar-open");
    $("#sidebar-toggle").setAttribute("aria-expanded",
      String(document.body.classList.contains("sidebar-open")));
  };
  $("#send-btn").onclick = () => streamChat();
  $("#stop-btn").onclick = stopChat;
  $("#attach-btn").onclick = () => $("#file-input").click();
  $("#hub-refresh").onclick = () => {
    state.localOnly = false;
    loadModels();
  };
  $("#hub-local").onclick = () => {
    state.localOnly = !state.localOnly;
    $("#hub-local").textContent = state.localOnly ? "Show all" : "Installed only";
    loadModels();
  };
  $("#hub-search").oninput = debounce(loadModels, 500);
  $("#hub-task").onchange = loadModels;
  $("#hub-sort").onchange = loadModels;
  $("#hub-deep").onchange = loadModels;
  $("#unload-btn").onclick = async () => {
    await api("/api/engine/unload", { method: "POST" });
    await loadStatus();
  };
  $("#set-save").onclick = saveSettings;
  $("#set-reload").onclick = async () => {
    await api("/api/engine/load", { method: "POST" });
    await loadStatus();
  };
  $("#set-theme").onchange = (e) => applyTheme(e.target.value);
  $("#img-go").onclick = generateImage;
  
  // Input enter key
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!state.streaming) streamChat();
    }
  });
}

/* ---------------- settings ---------------- */
async function fillSettings() {
  try {
    const d = await api("/api/settings");
    const s = d.settings;
    $("#set-model").value = s.model || "";
    $("#set-vision").value = s.vision_model || "";
    $("#set-embedding").value = s.embedding_model || "";
    $("#set-image").value = s.image_model || "";
    $("#set-tokens").value = s.max_new_tokens || 768;
    $("#set-temp").value = s.temperature || 0.7;
    $("#set-topp").value = s.top_p || 0.9;
    $("#set-topk").value = s.top_k || 50;
    $("#set-rep").value = s.repetition_penalty || 1.08;
    $("#set-history").value = s.history_messages || 12;
    $("#set-device").value = s.device || "auto";
    $("#set-dtype").value = s.dtype || "auto";
    $("#set-vismode").value = s.vision_mode || "auto";
    $("#set-rounds").value = s.max_tool_rounds || 3;
    $("#set-router").checked = s.agent_router ?? true;
    $("#set-stream").checked = s.stream ?? true;
    $("#set-autoanalyze").checked = s.vision_auto_analyze ?? true;
    $("#set-chunk").value = s.chunk_size || 900;
    $("#set-overlap").value = s.chunk_overlap || 120;
    $("#set-topkrag").value = s.top_k_rag || 4;
    $("#set-budget").value = s.context_budget || 6000;
    if (d.hf_token_set) $("#set-token").value = "********";
  } catch (_) {}
}
async function saveSettings() {
  const patch = {
    model: $("#set-model").value.trim(),
    vision_model: $("#set-vision").value.trim(),
    embedding_model: $("#set-embedding").value.trim(),
    image_model: $("#set-image").value.trim(),
    max_new_tokens: parseInt($("#set-tokens").value),
    temperature: parseFloat($("#set-temp").value),
    top_p: parseFloat($("#set-topp").value),
    top_k: parseInt($("#set-topk").value),
    repetition_penalty: parseFloat($("#set-rep").value),
    history_messages: parseInt($("#set-history").value),
    device: $("#set-device").value,
    dtype: $("#set-dtype").value,
    vision_mode: $("#set-vismode").value,
    max_tool_rounds: parseInt($("#set-rounds").value),
    agent_router: $("#set-router").checked,
    stream: $("#set-stream").checked,
    vision_auto_analyze: $("#set-autoanalyze").checked,
    chunk_size: parseInt($("#set-chunk").value),
    chunk_overlap: parseInt($("#set-overlap").value),
    top_k_rag: parseInt($("#set-topkrag").value),
    context_budget: parseInt($("#set-budget").value),
  };
  const tokenVal = $("#set-token").value;
  if (tokenVal && tokenVal !== "********") {
    patch.hf_token = tokenVal;
  }
  try {
    await api("/api/settings", {
      method: "POST",
      body: JSON.stringify(patch),
    });
    $("#set-msg").textContent = "Settings saved.";
    setTimeout(() => $("#set-msg").textContent = "", 3000);
    await loadStatus();
  } catch (e) {
    $("#set-msg").textContent = "Error: " + e.message;
  }
}

/* ---------------- resources & usage ---------------- */
function setMetric(id, text, level) {
  const element = $(id);
  element.textContent = text;
  element.dataset.level = level;
}
function metricLevel(value, warning = 60, danger = 80) {
  if (!Number.isFinite(value)) return "unknown";
  if (value >= danger) return "danger";
  if (value >= warning) return "warn";
  return "good";
}
async function loadResources() {
  try {
    const data = await api("/api/system/resources");
    const gpu = data.gpu || {};
    const toMetric = (value) => value == null || value === "" ? NaN : Number(value);
    const cpu = toMetric(data.cpu_percent);
    const ram = toMetric(data.ram_percent);
    const gpuUtil = toMetric(gpu.utilization_percent);
    const gpuMemory = toMetric(gpu.memory_percent);
    const gpuTemp = toMetric(gpu.temperature_c);
    setMetric("#metric-cpu", `CPU ${Number.isFinite(cpu) ? `${cpu.toFixed(0)}%` : "—"}`,
      metricLevel(cpu));
    setMetric("#metric-ram", `RAM ${Number.isFinite(ram) ? `${ram.toFixed(0)}%` : "—"}`,
      metricLevel(ram));
    setMetric("#metric-gpu-util",
      `GPU ${gpu.available && Number.isFinite(gpuUtil) ? `${gpuUtil.toFixed(0)}%` : "—"}`,
      metricLevel(gpuUtil));
    setMetric("#metric-gpu-vram",
      `GPU RAM ${gpu.available && Number.isFinite(gpuMemory)
        ? `${gpuMemory.toFixed(0)}% (${gpu.used_mb}/${gpu.total_mb} MB)` : "—"}`,
      metricLevel(gpuMemory));
    setMetric("#metric-gpu-temp",
      `GPU Temp ${gpu.available && Number.isFinite(gpuTemp) ? `${gpuTemp.toFixed(0)}°C` : "—"}`,
      metricLevel(gpuTemp, 70, 85));
  } catch (e) {
    console.warn("Could not refresh system resource metrics", e);
  }
}
async function loadUsage() {
  try {
    const d = await api("/api/usage");
    const projects = d.projects?.total ?? 0;
    const chats = d.chats?.total ?? 0;
    const messages = d.chats?.total_messages ?? 0;
    $("#stat-projects").textContent = `${projects} projects`;
    $("#stat-chats").textContent = `${chats} chats · ${messages} messages`;
    $("#stat-model").textContent = `model: ${d.model || "-"}`;
    $("#runtime-usage").textContent =
      `Projects ${projects} · Chat windows ${chats} · Messages ${messages}`;
  } catch (e) {
    console.warn("Could not refresh project usage", e);
  }
}
function startPolling() {
  if (state.polling) return;
  state.polling = true;
  setInterval(async () => {
    if (!state.user) return;
    try {
      await Promise.all([loadResources(), loadUsage(), loadStatus()]);
      if (state.tab === "models") {
        const jobs = await api("/api/models/jobs");
        if (jobs.jobs.some((job) => job.status === "downloading")) renderModels();
      }
    } catch (error) {
      console.warn("Background refresh failed", error);
    }
  }, 5000);
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// Boot
window.onload = boot;