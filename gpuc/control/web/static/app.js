// The dashboard page: fetch the same documents the CLI prints with --json and
// draw them. No state of its own beyond what is on screen; every action is a
// POST to an endpoint that runs the CLI's code.

"use strict";

const REFRESH_MS = 15000;
const LOG_FOLLOW_MS = 5000;
const LOG_LINES = 300;

const state = {
  auto: true,
  timer: null,
  loading: false,
  gatheredAt: null,
  log: null, // {jobId, host, timer}
};

// -- DOM helpers ------------------------------------------------------------

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function replace(node, ...children) {
  clear(node);
  node.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

// -- formatting: the same words `gpuc status` uses ---------------------------

function fmtDuration(seconds) {
  seconds = Math.max(0, seconds);
  const minutes = Math.round(seconds / 60);
  if (minutes === 0) return `${Math.round(seconds)}s`;
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (hours < 24) return `${hours}h${String(rest).padStart(2, "0")}m`;
  return `${Math.floor(hours / 24)}d${String(hours % 24).padStart(2, "0")}h`;
}

function fmtAge(stamp) {
  if (!stamp) return "age unknown";
  const seconds = (Date.now() - Date.parse(stamp)) / 1000;
  if (Number.isNaN(seconds)) return "age unknown";
  if (seconds < 0) return "just now";
  for (const [unit, size] of [["d", 86400], ["h", 3600], ["m", 60]]) {
    if (seconds >= size) return `${Math.floor(seconds / size)}${unit} ago`;
  }
  return `${Math.floor(seconds)}s ago`;
}

function fmtMinutes(job) {
  return job.elapsed_s === null ? "--" : `${(job.elapsed_s / 60).toFixed(1)}m`;
}

function fmtUtil(job) {
  return job.util === null ? "--" : `${Math.round(job.util)}%`;
}

function fmtEta(job) {
  if (job.eta_s === null) {
    return job.estimated_runtime_min === null
      ? ""
      : `est ${fmtDuration(job.estimated_runtime_min * 60)} total`;
  }
  const source = job.progress_pct ? `${Math.round(job.progress_pct)}%` : "est";
  return job.eta_s < 0 ? `overdue (${source})` : `${fmtDuration(job.eta_s)} (${source})`;
}

function fmtStarts(job) {
  if (job.starts_in_s === null || job.starts_in_s === undefined) return "";
  return job.starts_in_s < 60 ? "now" : `in ~${fmtDuration(job.starts_in_s)}`;
}

function fmtBytes(n) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function short(commit) {
  return commit ? commit.slice(0, 12) : "unknown";
}

// -- the API ----------------------------------------------------------------

async function api(path, options) {
  const response = await fetch(path, options);
  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("not logged in");
  }
  let document_;
  try {
    document_ = await response.json();
  } catch (err) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  if (document_.error) throw new Error(document_.error);
  return document_;
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

// -- notices ----------------------------------------------------------------

function notify(message, level) {
  const box = document.getElementById("notices");
  const note = el("div", { class: `notice ${level || ""}` }, message);
  box.append(note);
  setTimeout(() => note.remove(), 12000);
}

// -- rendering --------------------------------------------------------------

function badge(text, level) {
  return el("span", { class: `badge ${level || ""}` }, text);
}

function jobLabel(job) {
  return el("span", {},
    job.name ? el("span", { class: "job-name" }, job.name, " ") : null,
    el("span", { class: "job-id" }, job.job_id),
    job.attempt > 1 ? el("span", { class: "muted" }, ` attempt ${job.attempt}`) : null,
  );
}

const LINK_LABELS = { s3: "S3", hf: "HF", wandb: "W&B", mirror: "log mirror" };

function links(job) {
  const items = (job.links || []).filter((l) => l.url);
  if (!items.length) return el("span", { class: "muted" }, "");
  return el("span", { class: "links" },
    items.map((link) => el("a", {
      href: link.url, target: "_blank", rel: "noopener",
      title: `${link.target}${link.path ? ` (${link.path})` : ""}`,
    }, LINK_LABELS[link.kind] || link.kind)),
  );
}

function logsButton(host, job) {
  return el("button", { type: "button", onclick: () => openLog(job.job_id, host.name) }, "Logs");
}

function cancelButton(host, job) {
  return el("button", {
    type: "button", class: "danger",
    onclick: async (event) => {
      if (!window.confirm(`Cancel ${job.name || job.job_id} on ${host.name}?`)) return;
      event.target.disabled = true;
      try {
        const result = await post(`/api/jobs/${encodeURIComponent(job.job_id)}/cancel`, { host: host.name });
        notify(`job ${job.job_id} on ${result.host}: ${result.status}`);
        await load();
      } catch (err) {
        notify(err.message, "bad");
        event.target.disabled = false;
      }
    },
  }, "Cancel");
}

function preemptButton(host, job) {
  return el("button", {
    type: "button",
    onclick: async (event) => {
      if (!window.confirm(`Stop ${job.name || job.job_id} on ${host.name} and queue it again? It re-runs from the start.`)) return;
      event.target.disabled = true;
      try {
        const result = await post(`/api/jobs/${encodeURIComponent(job.job_id)}/preempt`, { host: host.name });
        for (const warning of result.warnings || []) notify(warning, "warn");
        notify(`job ${job.job_id} on ${result.host}: ${result.status}; it will be queued again at priority ${result.priority}`);
        await load();
      } catch (err) {
        notify(err.message, "bad");
        event.target.disabled = false;
      }
    },
  }, "Preempt");
}

function priorityControl(host, job) {
  const input = el("input", { type: "number", min: 0, max: 99, value: job.priority ?? 50, "aria-label": "priority" });
  const button = el("button", {
    type: "button",
    onclick: async () => {
      const priority = Number.parseInt(input.value, 10);
      if (Number.isNaN(priority)) return;
      button.disabled = true;
      try {
        const result = await post(`/api/jobs/${encodeURIComponent(job.job_id)}/reorder`, { host: host.name, priority });
        for (const warning of result.warnings || []) notify(warning, "warn");
        notify(`job ${job.job_id} moved to priority ${result.priority}`);
        await load();
      } catch (err) {
        notify(err.message, "bad");
        button.disabled = false;
      }
    },
  }, "Set");
  return el("span", { class: "actions" }, input, button);
}

function estimateButton(host, job) {
  return el("button", {
    type: "button",
    onclick: async () => {
      const raw = window.prompt(`Estimated runtime for ${job.name || job.job_id}, in minutes (blank clears it):`,
        job.estimated_runtime_min ?? "");
      if (raw === null) return;
      const body = { host: host.name };
      if (raw.trim() === "") {
        body.clear = true;
      } else {
        const minutes = Number(raw);
        if (!Number.isFinite(minutes)) {
          notify(`${raw.trim()} is not a number of minutes`, "bad");
          return;
        }
        body.minutes = minutes;
      }
      try {
        const result = await post(`/api/jobs/${encodeURIComponent(job.job_id)}/estimate`, body);
        for (const warning of result.warnings || []) notify(warning, "warn");
        notify(result.estimated_runtime_min === null
          ? `job ${job.job_id} no longer estimates a runtime`
          : `job ${job.job_id} now estimates ${result.estimated_runtime_min} min`);
        await load();
      } catch (err) {
        notify(err.message, "bad");
      }
    },
  }, "Estimate");
}

function table(headers, rows) {
  return el("table", {},
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", { class: h.num ? "num" : null }, h.text)))),
    el("tbody", {}, rows),
  );
}

function gpuTable(host) {
  if (!host.gpus.length) return el("p", { class: "empty" }, "no GPUs");
  const rows = host.gpus.map((gpu) => {
    if (gpu.available === false) {
      return el("tr", {},
        el("td", { class: "gpu-state" }, `[${gpu.owned_as}]`),
        el("td", {}, badge("UNAVAILABLE", "bad"), " nvidia-smi does not report this card, so nothing is dispatched to it"),
        el("td", {}), el("td", {}),
      );
    }
    return el("tr", {},
      el("td", { class: "gpu-state" }, `[${gpu.index ?? "?"}]`),
      el("td", {}, gpu.busy_job ? badge("busy", "warn") : badge("free", "good")),
      el("td", {}, gpu.name || "?", gpu.vram_mib ? ` ${Math.round(gpu.vram_mib / 1024)} GB` : ""),
      el("td", { class: "job-id" }, gpu.busy_job || ""),
    );
  });
  return table([{ text: "card" }, { text: "state" }, { text: "model" }, { text: "held by" }], rows);
}

function gpuLabels(host, job) {
  if (!job.gpus.length) return "none";
  return job.gpus.map((uuid) => {
    const card = host.gpus.find((g) => g.uuid === uuid);
    return card && card.index !== null && card.index !== undefined ? String(card.index) : uuid;
  }).join(",");
}

function runningTable(host) {
  if (!host.running.length) return null;
  const rows = host.running.map((job) => el("tr", { class: job.suspect ? "suspect" : null },
    el("td", {}, jobLabel(job)),
    el("td", {}, job.phase || "-"),
    el("td", { class: "num" }, fmtMinutes(job)),
    el("td", { class: "num" }, fmtUtil(job)),
    el("td", { class: "mono" }, gpuLabels(host, job)),
    el("td", {}, fmtEta(job), job.progress_error ? el("span", { class: "muted", title: job.progress_error }, " (progress error)") : null),
    el("td", {}, links(job)),
    el("td", { class: "actions" }, logsButton(host, job), estimateButton(host, job), preemptButton(host, job), cancelButton(host, job)),
  ));
  return el("div", {},
    el("div", { class: "section-label" }, "running"),
    table([{ text: "job" }, { text: "phase" }, { text: "elapsed", num: true }, { text: "util", num: true },
      { text: "gpu" }, { text: "eta" }, { text: "links" }, { text: "" }], rows),
  );
}

function queuedTable(host) {
  if (!host.queued.length) return null;
  const rows = host.queued.map((job) => el("tr", {},
    el("td", {}, jobLabel(job)),
    el("td", {}, priorityControl(host, job)),
    el("td", {}, job.estimated_runtime_min === null ? "" : `est ${fmtDuration(job.estimated_runtime_min * 60)}`),
    el("td", {}, fmtStarts(job)),
    el("td", {}, links(job)),
    el("td", { class: "actions" }, logsButton(host, job), estimateButton(host, job), cancelButton(host, job)),
  ));
  return el("div", {},
    el("div", { class: "section-label" }, "queued (lower priority dispatches first)"),
    table([{ text: "job" }, { text: "priority" }, { text: "estimate" }, { text: "starts" }, { text: "links" }, { text: "" }], rows),
  );
}

function finishedTable(host) {
  if (!host.finished.length) return null;
  const rows = host.finished.map((job) => {
    const detail = job.reason || "";
    let level = "good";
    if (job.status === "failed") level = "bad";
    else if (job.status === "cancelled") level = "warn";
    let flag = null;
    if (job.outputs_lost && job.outputs_pending) flag = badge("OUTPUTS LOST", "bad");
    else if (job.outputs_pending) flag = badge("outputs not uploaded", "warn");
    return el("tr", {},
      el("td", {}, jobLabel(job)),
      el("td", {}, badge(job.status, level), detail ? ` ${detail}` : ""),
      el("td", {}, fmtAge(job.ended_at)),
      el("td", {}, job.progress_pct !== null ? `${Math.round(job.progress_pct)}%` : ""),
      el("td", {}, flag, job.workdir_bytes ? el("span", { class: "muted" }, ` workdir ${fmtBytes(job.workdir_bytes)}`) : null),
      el("td", {}, links(job)),
      el("td", { class: "actions" }, logsButton(host, job)),
    );
  });
  return el("div", {},
    el("div", { class: "section-label" }, "finished"),
    table([{ text: "job" }, { text: "result" }, { text: "ended" }, { text: "progress" }, { text: "outputs" }, { text: "links" }, { text: "" }], rows),
  );
}

function hostHeader(host, entry) {
  const target = host.target || (host.kind === "local" ? "this machine" : "");
  let stateBadge;
  if (host.pod_gone) stateBadge = badge("POD GONE", "bad");
  else if (!host.reachable) stateBadge = badge("UNREACHABLE", "bad");
  else if (host.dispatcher.alive) stateBadge = badge(`dispatcher ${Math.round(host.dispatcher.heartbeat_age_s)}s ago`, "good");
  else stateBadge = badge("dispatcher DOWN", "bad");
  const free = host.gpus.filter((g) => g.available !== false && !g.busy_job).length;
  const owned = host.gpus.filter((g) => g.available !== false).length;
  const meta = el("div", { class: "meta" },
    host.reachable ? el("span", {}, host.gpus.length ? `gpus ${free}/${owned} free` : "no GPUs") : null,
    entry && entry.driver_version ? el("span", {}, `driver ${entry.driver_version}`) : null,
    el("span", {}, `pkg ${short(host.pkg_commit)}`),
    entry && entry.s3_prefix ? el("span", { class: "mono" }, `mirror ${entry.s3_prefix}`) : null,
    entry && entry.persistent_root ? el("span", { class: "mono" }, `root ${entry.persistent_root}`) : null,
    entry && entry.retention_days !== null && entry.retention_days !== undefined ? el("span", {}, `retention ${entry.retention_days}d`) : null,
    entry && entry.workdir_days !== null && entry.workdir_days !== undefined ? el("span", {}, `workdirs ${entry.workdir_days}d`) : null,
    host.kind === "runpod" && entry ? el("span", {}, `idle ${entry.idle_minutes}m`, entry.ttl_hours !== null && entry.ttl_hours !== undefined ? `, ttl ${entry.ttl_hours}h` : "") : null,
  );
  const flags = el("div", { class: "flags" },
    host.draining ? badge("DRAINING", "warn") : null,
    host.paused ? badge(`PAUSED (low-util); resume with gpuc host resume ${host.name}`, "warn") : null,
    host.pod && host.pod.past_ttl ? badge("PAST TTL", "warn") : null,
  );
  return [
    el("header", {},
      el("h3", {}, host.name),
      badge(host.kind, "kind"),
      target ? el("span", { class: "muted mono" }, target) : null,
      stateBadge,
    ),
    meta,
    flags.childElementCount ? flags : null,
  ];
}

function podLine(host) {
  const pod = host.pod;
  if (!pod) return null;
  const util = host.provider_util && host.provider_util.length ? host.provider_util.map((u) => `${u}%`).join(",") : "--";
  return el("div", { class: "meta" },
    el("span", { class: "mono" }, `pod ${pod.id}`),
    el("span", {}, pod.status),
    el("span", {}, pod.gpu_name || "?"),
    el("span", {}, `$${pod.cost_usd_hr.toFixed(3)}/h`),
    el("span", {}, `cuda ${pod.cuda_version || "?"}`),
    el("span", {}, pod.age_s === null ? "age ?" : `age ${Math.round(pod.age_s / 60)}m`),
    el("span", {}, `provider util ${util}`),
  );
}

function hostCard(host, entry) {
  const card = el("article", { class: "host" }, hostHeader(host, entry));
  for (const error of host.errors) card.append(el("div", { class: "notice bad" }, error));
  if (!host.reachable) {
    if (!host.pod_gone) card.append(el("p", { class: "muted" }, `try: gpuc host probe ${host.name}`));
    const pod = podLine(host);
    if (pod) card.append(pod);
    return card;
  }
  card.append(gpuTable(host));
  const pod = podLine(host);
  if (pod) card.append(pod);
  const sections = [runningTable(host), queuedTable(host), finishedTable(host)].filter(Boolean);
  if (!sections.length) card.append(el("p", { class: "empty" }, "idle; nothing queued, running or finished"));
  card.append(...sections);
  const leftover = host.finished.reduce((sum, job) => sum + (job.workdir_bytes || 0), 0);
  if (leftover > 1 << 30) {
    card.append(el("p", { class: "muted" },
      `${fmtBytes(leftover)} still in finished job workdirs; free it with: `,
      el("code", {}, `gpuc clean --host ${host.name} --all-finished`)));
  }
  return card;
}

function renderHosts(status, hosts) {
  const list = document.getElementById("host-list");
  const entries = new Map((hosts.hosts || []).map((h) => [h.name, h]));
  if (!status.hosts.length) {
    replace(list, el("p", { class: "empty" }, "no hosts registered. Add one: gpuc host add local --gpus 0"));
    return;
  }
  replace(list, status.hosts.map((host) => hostCard(host, entries.get(host.name))));
}

function renderConfig(config, version) {
  const body = document.getElementById("config-body");
  const rows = Object.entries(config.settings).map(([key, value]) => el("tr", {},
    el("td", {}, key),
    el("td", { class: "mono" }, value === null ? el("span", { class: "muted" }, "unset") : String(value)),
  ));
  const versionText = version
    ? `gpuc ${version.version} · commit ${short(version.commit)} [${version.source}]${version.dirty ? " +uncommitted" : ""}`
    : "";
  replace(body,
    el("div", { class: "meta" },
      el("span", { class: "mono" }, `config file ${config.config_file}${config.config_file_exists ? "" : " (does not exist; using defaults)"}`),
      el("span", { class: "mono" }, `state dir ${config.state_dir}`),
      versionText ? el("span", {}, versionText) : null,
    ),
    table([{ text: "setting" }, { text: "value" }], rows),
    config.notes.map((note) => el("p", { class: "muted" }, `note: ${note}`)),
  );
}

// -- loading ------------------------------------------------------------------

async function load() {
  if (state.loading) return;
  state.loading = true;
  const updated = document.getElementById("updated");
  updated.textContent = "refreshing…";
  try {
    const [status, hosts, config, version] = await Promise.all([
      api("/api/status"), api("/api/hosts"), api("/api/config"), api("/api/version"),
    ]);
    // A refresh must not pull a half-typed priority out from under someone.
    if (!editingInHosts()) renderHosts(status, hosts);
    renderConfig(config, version);
    for (const error of status.errors) notify(error, "bad");
    state.gatheredAt = Date.now();
    tickUpdated();
  } catch (err) {
    updated.textContent = `refresh failed: ${err.message}`;
  } finally {
    state.loading = false;
  }
}

function editingInHosts() {
  const active = document.activeElement;
  return active instanceof HTMLInputElement && active.closest("#host-list") !== null;
}

function tickUpdated() {
  if (state.gatheredAt === null) return;
  document.getElementById("updated").textContent = `updated ${fmtAge(new Date(state.gatheredAt).toISOString())}`;
}

function schedule() {
  if (state.timer) clearInterval(state.timer);
  state.timer = state.auto ? setInterval(load, REFRESH_MS) : null;
}

// -- the log panel ------------------------------------------------------------

async function fetchLog() {
  const log = state.log;
  if (!log || log.busy) return;
  log.busy = true;
  const text = document.getElementById("log-text");
  try {
    const result = await api(`/api/jobs/${encodeURIComponent(log.jobId)}/logs?lines=${LOG_LINES}&host=${encodeURIComponent(log.host)}`);
    // The panel may have moved on to another job while this was in flight.
    if (state.log !== log) return;
    document.getElementById("log-where").textContent = `${result.source}: ${result.location || ""}`;
    const atBottom = text.scrollTop + text.clientHeight >= text.scrollHeight - 4;
    text.textContent = result.lines.join("\n") + (result.notes.length ? `\n\n[${result.notes.join("; ")}]` : "");
    if (atBottom || log.firstLoad) text.scrollTop = text.scrollHeight;
    log.firstLoad = false;
  } catch (err) {
    if (state.log === log) text.textContent = `could not read the log: ${err.message}`;
  } finally {
    log.busy = false;
    scheduleFollow(log);
  }
}

function openLog(jobId, host) {
  closeLog();
  state.log = { jobId, host, timer: null, busy: false, firstLoad: true };
  document.getElementById("log-title").textContent = `log \u00b7 ${jobId} \u00b7 ${host}`;
  document.getElementById("log-text").textContent = "loading\u2026";
  document.getElementById("log-panel").hidden = false;
  document.body.classList.add("log-open");
  fetchLog();
}

// One fetch at a time, the next one scheduled only once the last has landed:
// a log tail is an ssh round trip, and an interval would stack them on a slow
// host and let an older answer overwrite a newer one.
function scheduleFollow(log) {
  log = log || state.log;
  if (!log || state.log !== log) return;
  if (log.timer) clearTimeout(log.timer);
  log.timer = document.getElementById("log-follow").checked ? setTimeout(fetchLog, LOG_FOLLOW_MS) : null;
}

function closeLog() {
  if (state.log && state.log.timer) clearTimeout(state.log.timer);
  state.log = null;
  document.getElementById("log-panel").hidden = true;
  document.body.classList.remove("log-open");
}

// -- wiring -------------------------------------------------------------------

document.getElementById("refresh").addEventListener("click", load);
document.getElementById("auto").addEventListener("change", (event) => {
  state.auto = event.target.checked;
  schedule();
});
document.getElementById("log-refresh").addEventListener("click", fetchLog);
document.getElementById("log-close").addEventListener("click", closeLog);
document.getElementById("log-follow").addEventListener("change", () => scheduleFollow());
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.auto) load();
});
setInterval(tickUpdated, 1000);
load();
schedule();
