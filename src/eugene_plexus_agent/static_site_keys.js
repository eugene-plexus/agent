// A person's own key for this job site (J14a, person-held-keys.md).
// Made here, in this browser, on the machine's loopback page: WebCrypto
// will not hand its private half to any script, this one included, and it
// never leaves this browser. Its public half is pinned at the machine by
// the agent. The page then signs, with it, exactly the text the site host
// gave for each change the person approves.
"use strict";
(() => {
  const DB = "eugene-plexus-site-keys";
  const STORE = "keys";

  function openDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async function store(mode, act) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const request = act(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(request ? request.result : undefined);
      tx.onerror = () => reject(tx.error);
    });
  }

  const allKeys = () => store("readonly", (s) => s.getAll());
  const putKey = (value) => store("readwrite", (s) => s.put(value));

  function base64(buffer) {
    let text = "";
    for (const byte of new Uint8Array(buffer)) text += String.fromCharCode(byte);
    return btoa(text);
  }

  async function keyId(raw) {
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", raw));
    return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
  }

  async function makeKey() {
    try {
      const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, false, ["sign", "verify"]);
      return { alg: "Ed25519", pair, raw: await crypto.subtle.exportKey("raw", pair.publicKey) };
    } catch (_) {
      // A browser without Ed25519 in WebCrypto: ECDSA on P-256.
      const pair = await crypto.subtle.generateKey(
        { name: "ECDSA", namedCurve: "P-256" },
        false,
        ["sign", "verify"],
      );
      return { alg: "ES256", pair, raw: await crypto.subtle.exportKey("raw", pair.publicKey) };
    }
  }

  async function sign(held, text) {
    const params = held.alg === "Ed25519" ? { name: "Ed25519" } : { name: "ECDSA", hash: "SHA-256" };
    const data = new TextEncoder().encode(text);
    return base64(await crypto.subtle.sign(params, held.privateKey, data));
  }

  function grouped(id) {
    return id.slice(0, 16).match(/.{4}/g).join(" ");
  }

  function say(element, text) {
    element.textContent = text;
  }

  async function post(path, csrf, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Eugene-Csrf": csrf },
      body: JSON.stringify(body),
      credentials: "same-origin",
      cache: "no-store",
    });
    let value = {};
    try {
      value = await response.json();
    } catch (_) {
      value = {};
    }
    return { ok: response.ok, status: response.status, value };
  }

  async function heldKey(pinned) {
    const held = await allKeys();
    return held.find((k) => pinned.includes(k.id)) || null;
  }

  function usable() {
    return window.isSecureContext && window.crypto && crypto.subtle && window.indexedDB;
  }

  // --- the link page: make and pin a key ------------------------------------------

  async function keyPanel(panel) {
    const csrf = panel.dataset.csrf;
    const pinned = JSON.parse(panel.dataset.keys || "[]");
    const state = panel.querySelector("[data-state]");
    const button = panel.querySelector("[data-make]");
    if (!usable()) {
      say(state, "This browser cannot keep a key here. Open this page in Chrome, Edge or Firefox.");
      return;
    }
    const held = await heldKey(pinned);
    if (held) {
      say(state, `This browser holds your key ${grouped(held.id)}.`);
      return;
    }
    say(
      state,
      pinned.length
        ? "This browser does not hold any of your keys. You can make one here too."
        : "You have no key here yet. Make one in this browser.",
    );
    button.hidden = false;
    button.addEventListener("click", async () => {
      button.disabled = true;
      say(state, "Making your key…");
      try {
        const made = await makeKey();
        const id = await keyId(made.raw);
        const answer = await post("/link/key", csrf, { alg: made.alg, publicKey: base64(made.raw) });
        if (!answer.ok || answer.value.id !== id) {
          say(state, answer.value.detail || "The key could not be added. Open this page again.");
          button.disabled = false;
          return;
        }
        await putKey({ id, alg: made.alg, privateKey: made.pair.privateKey, publicKey: made.pair.publicKey });
        window.location.assign("/link/approve");
      } catch (error) {
        say(state, `The key could not be made (${error && error.name ? error.name : "error"}).`);
        button.disabled = false;
      }
    });
  }

  // --- the approval page: sign what the site host holds ------------------------------

  // What each kind of item is, and what its button does (J14b: a call from
  // Workbench runs once signed; a window lets the tools a person's rules
  // allow run without a signature each, for an hour).
  const KINDS = {
    "rules.confirm": ["Approve this machine's rules", "Approve these rules"],
    call: ["A call from Workbench, waiting for your signature", "Sign: let it run"],
    "window.open": ["Workbench asks for a window", "Sign: open the window"],
  };

  function card(item) {
    const box = document.createElement("section");
    box.className = "held";
    const [heading, action] = KINDS[item.action] || ["A change from Workbench", "Approve"];
    const title = document.createElement("h2");
    title.textContent = heading;
    box.append(title);
    const list = document.createElement("div");
    for (const line of item.words || []) {
      const p = document.createElement("p");
      p.textContent = line;
      if (item.action === "call") {
        // The call exactly, as this machine will run it.
        p.style.whiteSpace = "pre-wrap";
        p.style.fontFamily = "ui-monospace, Consolas, monospace";
      }
      list.append(p);
    }
    box.append(list);
    const approve = document.createElement("button");
    approve.textContent = action;
    approve.dataset.approve = item.id;
    box.append(approve);
    if (item.action !== "rules.confirm") {
      const reject = document.createElement("button");
      reject.textContent = "Turn down";
      reject.dataset.reject = item.id;
      box.append(" ", reject);
    }
    return box;
  }

  async function approvals(panel) {
    const csrf = panel.dataset.csrf;
    const pinned = JSON.parse(panel.dataset.keys || "[]");
    const state = panel.querySelector("[data-state]");
    const list = panel.querySelector("[data-items]");
    if (!usable()) {
      say(state, "This browser cannot keep a key here. Open this page in Chrome, Edge or Firefox.");
      return;
    }
    const held = await heldKey(pinned);
    if (!held) {
      say(state, "This browser holds no key of yours. Make one on the link page first.");
      return;
    }

    async function load() {
      list.replaceChildren();
      const response = await fetch(`/link/approve/items?key=${held.id}`, {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (!response.ok) {
        let detail = "";
        try {
          detail = (await response.json()).detail || "";
        } catch (_) {
          detail = "";
        }
        say(state, detail || "What is waiting could not be read. Open this page again.");
        return [];
      }
      const value = await response.json();
      const items = value.items || [];
      say(
        state,
        items.length
          ? `Signing with your key ${grouped(held.id)}. Read each change before you approve it.`
          : "Nothing is waiting for your approval.",
      );
      for (const item of items) list.append(card(item));
      return items;
    }

    let items = await load();
    let busy = false;
    // A call Workbench sends while this page is open shows without a reload.
    setInterval(async () => {
      if (busy || document.hidden) return;
      const before = items.map((i) => i.id).join(",");
      const fresh = await fetch(`/link/approve/items?key=${held.id}`, {
        credentials: "same-origin",
        cache: "no-store",
      }).catch(() => null);
      if (!fresh || !fresh.ok || busy) return;
      const value = await fresh.json();
      if ((value.items || []).map((i) => i.id).join(",") !== before) items = await load();
    }, 5000);
    list.addEventListener("click", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) return;
      const approveId = target.dataset.approve;
      const rejectId = target.dataset.reject;
      const id = approveId || rejectId;
      if (!id) return;
      busy = true;
      for (const button of list.querySelectorAll("button")) button.disabled = true;
      let answer;
      if (approveId) {
        const item = items.find((i) => i.id === approveId);
        if (!item || !item.envelope) {
          items = await load();
          return;
        }
        const signature = await sign(held, item.envelope);
        answer = await post(`/link/approve/items/${encodeURIComponent(id)}`, csrf, {
          envelope: item.envelope,
          key: held.id,
          signature,
        });
      } else {
        answer = await post(`/link/approve/items/${encodeURIComponent(id)}/reject`, csrf, {});
      }
      items = await load();
      busy = false;
      if (!answer.ok || (answer.value.status && answer.value.status !== "done")) {
        say(state, answer.value.message || answer.value.detail || "That did not go through.");
      }
    });
  }

  const panel = document.getElementById("site-key");
  if (panel) keyPanel(panel);
  const approve = document.getElementById("site-approve");
  if (approve) approvals(approve);
})();
