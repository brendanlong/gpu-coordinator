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
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function replace(node, ...children) {
  node.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
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

function fmtElapsed(job) {
  return job.elapsed_s === null ? "--" : fmtDuration(job.elapsed_s);
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

function fmtWait(seconds) {
  return seconds < 60 ? "now" : `in ~${fmtDuration(seconds)}`;
}

function fmtStarts(job) {
  if (job.starts_in_s === null || job.starts_in_s === undefined) return "";
  return fmtWait(job.starts_in_s);
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

// The line `gpuc submit` and `gpuc reorder` print about where the job landed.
function queueNote(result) {
  if (result.dispatched) return "dispatched already; it is running now";
  if (result.queue_position === null || result.queue_position === undefined) return "";
  const starts = result.starts_in_s === null
    ? `start time unknown (${result.starts_unknown})`
    : `starts ${fmtWait(result.starts_in_s)}`;
  return `position ${result.queue_position} of ${result.queue_length}; ${starts}`;
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

function jobPath(job, action) {
  return `/api/jobs/${encodeURIComponent(job.job_id)}/${action}`;
}

// -- notices ----------------------------------------------------------------

function notify(message, level) {
  const box = document.getElementById("notices");
  const note = el("div", { class: `notice ${level || ""}` }, message);
  box.append(note);
  setTimeout(() => note.remove(), 12000);
}

// One shape for every action button: disable it, POST, report what the CLI
// would have printed, and redraw. A failure re-enables the button so the
// action can be retried without waiting for the next refresh.
async function act(button, path, body, describe) {
  button.disabled = true;
  try {
    const result = await post(path, body);
    for (const warning of result.warnings || []) notify(warning, "warn");
    notify(describe(result));
    await load();
  } catch (err) {
    notify(err.message, "bad");
    button.disabled = false;
  }
}

// -- rendering --------------------------------------------------------------

function badge(text, level, title) {
  return el("span", { class: `badge ${level || ""}`, title }, text);
}

function jobLabel(job) {
  return el("span", {},
    job.name ? el("span", { class: "job-name" }, job.name, " ") : null,
    el("span", { class: "job-id" }, job.job_id),
    job.attempt > 1 ? el("span", { class: "muted" }, ` attempt ${job.attempt}`) : null,
    // Only while it is queued or running: on a finished job it would read as
    // something that happened to it rather than something it allows.
    job.auto_preempt && (job.status === "queued" || job.status === "running")
      ? el("span", { class: "muted", title: "stopped and queued again whenever that lets a more important job start" }, " auto-preempt")
      : null,
  );
}

const LINK_LABELS = { s3: "S3", hf: "HF", wandb: "W&B", mirror: "log mirror" };

function links(job) {
  const items = (job.links || []).filter((l) => l.url);
  if (!items.length) return null;
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

// `gpuc host remove`: forget a host here, and nothing else. Offered only on
// a card nothing answers from, which is where the CLI's own hint points.
function forgetButton(host) {
  const button = el("button", {
    type: "button", class: "danger",
    onclick: () => {
      if (!window.confirm(`Forget host ${host.name} here? Nothing on it changes.`)) return;
      act(button, `/api/hosts/${encodeURIComponent(host.name)}/remove`, {},
        (r) => [`removed host ${r.host}`, ...r.notes].join("; "));
    },
  }, "Forget host");
  return button;
}

function cancelButton(host, job) {
  const button = el("button", {
    type: "button", class: "danger",
    onclick: () => {
      if (!window.confirm(`Cancel ${job.name || job.job_id} on ${host.name}?`)) return;
      act(button, jobPath(job, "cancel"), { host: host.name },
        (r) => `job ${job.job_id} on ${r.host}: ${r.status}`);
    },
  }, "Cancel");
  return button;
}

function preemptButton(host, job) {
  const button = el("button", {
    type: "button",
    onclick: () => {
      if (!window.confirm(`Stop ${job.name || job.job_id} on ${host.name} and queue it again? It re-runs from the start.`)) return;
      act(button, jobPath(job, "preempt"), { host: host.name },
        (r) => `job ${job.job_id} on ${r.host}: ${r.status}`
          + (r.priority === null ? "" : `; it will be queued again at priority ${r.priority}`));
    },
  }, "Preempt");
  return button;
}

function priorityControl(host, job) {
  const input = el("input", { type: "number", min: 0, max: 99, value: job.priority ?? 50, "aria-label": "priority" });
  const button = el("button", {
    type: "button",
    onclick: () => {
      const priority = Number.parseInt(input.value, 10);
      if (Number.isNaN(priority)) return;
      act(button, jobPath(job, "reorder"), { host: host.name, priority }, (r) => {
        const note = queueNote(r);
        return `job ${job.job_id} on ${r.host} moved to priority ${r.priority}${note ? `; ${note}` : ""}`;
      });
    },
  }, "Reorder");
  return el("span", { class: "actions" }, input, button);
}

function estimateButton(host, job) {
  const button = el("button", {
    type: "button",
    onclick: () => {
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
      act(button, jobPath(job, "estimate"), body, (r) => (r.estimated_runtime_min === null
        ? `job ${job.job_id} on ${r.host} no longer estimates a runtime`
        : `job ${job.job_id} on ${r.host} now estimates ${r.estimated_runtime_min} min`));
    },
  }, "Estimate");
  return button;
}

function table(headers, rows) {
  return el("table", {},
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", { class: h.num ? "num" : null, title: h.title }, h.text)))),
    el("tbody", {}, rows),
  );
}

function model(gpu) {
  return [gpu.name || "?", gpu.vram_mib ? ` ${Math.round(gpu.vram_mib / 1024)} GB` : ""];
}

function gpuRow(index, state, model) {
  return el("tr", {}, el("td", { class: "gpu-index" }, `[${index ?? "?"}]`), el("td", {}, state), el("td", {}, model));
}

function missingRow(entry, as, what) {
  return gpuRow(entry[as], [badge("UNAVAILABLE", "bad"), ` nvidia-smi does not report this card, so nothing is ${what} it`], "");
}

// `?` and not `0` for a reading the host could not take: that card is out
// *because* nothing is known about it, and "0 MiB, 0% util" beside IN USE
// reads as a bug.
function reading(v) {
  return v === null || v === undefined ? "?" : Math.round(v);
}

function gpuTable(host) {
  const shared = host.shared_gpus || [];
  if (!host.gpus.length && !shared.length) return el("p", { class: "empty" }, "no GPUs");
  // No holder column: the running table below names each job's cards.
  const rows = host.gpus.map((gpu) => (gpu.available === false
    ? missingRow(gpu, "owned_as", "dispatched to")
    : gpuRow(gpu.index, gpu.busy_job ? badge("busy", "warn") : badge("free", "good"), model(gpu))));
  // Shared cards are somebody else's, and `IN USE` is theirs, not ours: the
  // numbers beside it are why a job that asked for one is still queued.
  for (const gpu of shared) {
    if (gpu.available === false) {
      rows.push(missingRow(gpu, "shared_as", "borrowed from"));
      continue;
    }
    let state;
    if (gpu.busy_job) state = badge("shared, busy", "warn");
    else if (gpu.unused) state = badge("shared, free", "good");
    else state = [badge("shared, IN USE", "bad"), el("span", { class: "muted" }, ` somebody else: ${reading(gpu.memory_mib)} MiB, ${reading(gpu.utilization_pct)}% util`)];
    rows.push(gpuRow(gpu.index, state, model(gpu)));
  }
  return table([{ text: "card" }, { text: "state" }, { text: "model" }], rows);
}

function gpuLabels(host, job) {
  if (!job.gpus.length) return "none";
  const cards = [...host.gpus, ...(host.shared_gpus || [])];
  return job.gpus.map((uuid) => {
    const card = cards.find((g) => g.uuid === uuid);
    return card && card.index !== null && card.index !== undefined ? String(card.index) : uuid;
  }).join(",");
}

// The GPU table above can show a shared card sitting free while a job below it
// stays queued, and the missing half of that is whether the job is allowed on
// it at all. Only said on a host that has shared cards: everywhere else it is
// a line about borrowing on a host that never borrows.
function borrowLabel(host, job) {
  if (!(host.shared_gpus || []).length) return null;
  // Dotted off rather than run on: ` needs 2 gpus owned cards only` reads as
  // one phrase about the two cards.
  if (job.use_shared === null || job.use_shared === undefined) {
    return el("span", { class: "muted", title: "this host did not report whether the job may borrow -- it could not read the spec" },
      " · borrowing unknown");
  }
  return job.use_shared
    ? el("span", { class: "muted", title: "may run on the host's shared cards, while nobody else is on them" }, " · may borrow")
    : el("span", { class: "muted", title: "waits for the host's own cards; submit with use_shared to let it borrow" }, " · owned cards only");
}

function section(label, headers, rows) {
  return el("div", {}, el("div", { class: "section-label" }, label), table(headers, rows));
}

function runningTable(host) {
  if (!host.running.length) return null;
  const rows = host.running.map((job) => el("tr", {},
    el("td", {}, jobLabel(job)),
    el("td", {}, job.phase || "-"),
    el("td", { class: "num" }, fmtElapsed(job)),
    el("td", { class: "num" }, fmtUtil(job)),
    el("td", { class: "mono" }, gpuLabels(host, job)),
    el("td", {}, fmtEta(job), job.progress_error ? el("span", { class: "muted", title: job.progress_error }, " (progress error)") : null),
    el("td", {}, links(job)),
    el("td", { class: "actions" }, logsButton(host, job), estimateButton(host, job), preemptButton(host, job), cancelButton(host, job)),
  ));
  return section("running", [{ text: "job" }, { text: "phase" }, { text: "elapsed", num: true }, { text: "util", num: true },
    { text: "gpu" }, { text: "eta" }, { text: "links" }, { text: "" }], rows);
}

function queuedTable(host) {
  if (!host.queued.length) return null;
  const rows = host.queued.map((job) => el("tr", {},
    el("td", {}, jobLabel(job),
      // The usual job wants one card; a job waiting for three is the answer to
      // "there is a card free, why is it still queued".
      job.gpus_requested > 1 ? el("span", { class: "muted" }, ` needs ${job.gpus_requested} gpus`) : null,
      borrowLabel(host, job)),
    el("td", {}, priorityControl(host, job)),
    el("td", {}, job.estimated_runtime_min === null ? "" : `est ${fmtDuration(job.estimated_runtime_min * 60)}`),
    el("td", {}, fmtStarts(job)),
    el("td", {}, links(job)),
    el("td", { class: "actions" }, logsButton(host, job), estimateButton(host, job), cancelButton(host, job)),
  ));
  return section("queued", [{ text: "job" }, { text: "priority", title: "lower dispatches first" }, { text: "estimate" },
    { text: "starts" }, { text: "links" }, { text: "" }], rows);
}

function finishedTable(host) {
  if (!host.finished.length) return null;
  const rows = host.finished.map((job) => {
    const reason = job.reason === job.status ? null : job.reason;
    const detail = reason || (job.exit_code ? `exit ${job.exit_code}` : "");
    // How far a job had got when it ended is the useful part of a failure.
    const progress = job.progress_pct !== null && job.status !== "succeeded" ? ` (${Math.round(job.progress_pct)}%)` : "";
    let level = "good";
    if (job.status === "failed") level = "bad";
    else if (job.status === "cancelled") level = "warn";
    let flag = null;
    if (job.outputs_lost && job.outputs_pending) flag = badge("OUTPUTS LOST", "bad");
    else if (job.outputs_pending) flag = badge("outputs not uploaded", "warn");
    return el("tr", {},
      el("td", {}, jobLabel(job)),
      el("td", {}, badge(job.status, level), detail ? ` ${detail}` : "", progress),
      el("td", {}, fmtAge(job.ended_at)),
      el("td", {}, flag, job.workdir_bytes ? el("span", { class: "muted" }, ` workdir ${fmtBytes(job.workdir_bytes)}`) : null),
      el("td", {}, links(job)),
      el("td", { class: "actions" }, logsButton(host, job)),
    );
  });
  return section("finished", [{ text: "job" }, { text: "result" }, { text: "ended" }, { text: "outputs" }, { text: "links" }, { text: "" }], rows);
}

function cardsSummary(host) {
  const available = host.gpus.filter((g) => g.available !== false);
  const shared = (host.shared_gpus || []).filter((g) => g.available !== false);
  // A host that owns nothing and borrows something is a real configuration,
  // and `no GPUs` above a list of shared cards contradicts itself.
  if (available.length) return `gpus ${available.filter((g) => !g.busy_job).length}/${available.length} free`;
  if (shared.length) return `shared ${shared.filter((g) => g.unused && !g.busy_job).length}/${shared.length} free, none owned`;
  return "no GPUs";
}

function hostHeader(host, entry) {
  const target = host.target || (host.kind === "local" ? "this machine" : "");
  let stateBadge;
  if (host.state === "gone") stateBadge = badge("GONE", "bad");
  else if (!host.reachable) stateBadge = badge("UNASKABLE", "bad");
  else if (host.dispatcher.alive) stateBadge = badge(`dispatcher ${Math.round(host.dispatcher.heartbeat_age_s)}s ago`, "good");
  else stateBadge = badge("dispatcher DOWN", "bad", "submit or bootstrap restarts it");
  const meta = el("div", { class: "meta" },
    host.reachable ? el("span", {}, cardsSummary(host)) : null,
    entry && entry.s3_prefix ? el("span", { class: "mono" }, `mirror ${entry.s3_prefix}`) : null,
    entry && entry.retention_days !== null && entry.retention_days !== undefined ? el("span", {}, `retention ${entry.retention_days}d`) : null,
    entry && entry.workdir_days !== null && entry.workdir_days !== undefined ? el("span", {}, `workdirs ${entry.workdir_days}d`) : null,
    host.kind === "rental" && entry ? el("span", {}, `idle ${entry.idle_minutes}m`) : null,
  );
  const flags = el("div", { class: "flags" },
    host.draining ? badge("DRAINING", "warn") : null,
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
    el("span", {}, pod.age_s === null ? "age ?" : `age ${fmtDuration(pod.age_s)}`),
    el("span", {}, `provider util ${util}`),
  );
}

function hostCard(host, entry) {
  const card = el("article", { class: "host" }, hostHeader(host, entry));
  for (const error of host.errors) card.append(el("div", { class: "notice bad" }, error));
  for (const warning of host.warnings || []) card.append(el("div", { class: "notice warn" }, warning));
  if (!host.reachable) {
    if (host.state === "unaskable" && !host.pod) card.append(el("p", { class: "muted" }, `try: gpuc host probe ${host.name}`));
    card.append(podLine(host) || []);
    card.append(el("p", {}, forgetButton(host)));
    return card;
  }
  card.append(gpuTable(host), podLine(host) || []);
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
  // Registry trouble persists until it is fixed, so it is drawn in place and
  // redrawn on every refresh rather than toasted again each time.
  replace(document.getElementById("errors"), status.errors.map((error) => el("div", { class: "notice bad" }, error)));
  if (!status.hosts.length) {
    replace(list, el("p", { class: "empty" }, "no hosts registered. Add one: gpuc host add local"));
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
  replace(body,
    el("div", { class: "meta" },
      el("span", { class: "mono" }, `config file ${config.config_file}${config.config_file_exists ? "" : " (does not exist; using defaults)"}`),
      el("span", { class: "mono" }, `state dir ${config.state_dir}`),
      el("span", {}, `gpuc ${version.version} · commit ${short(version.commit)} [${version.source}]${version.dirty ? " +uncommitted" : ""}`),
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
    state.gatheredAt = status.gathered_at;
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
  document.getElementById("updated").textContent = `updated ${fmtAge(state.gatheredAt)}`;
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
  document.getElementById("log-title").textContent = `log · ${jobId} · ${host}`;
  document.getElementById("log-text").textContent = "loading…";
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
