const SHELL = "edgeagent-shell-v1", RUNTIME = "edgeagent-data-v1";
const FILES = ["./", "index.html", "app.js", "style.css", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png"];
self.addEventListener("install", e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES))); self.skipWaiting(); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => ![SHELL, RUNTIME].includes(k)).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (u.pathname.startsWith("/v1/") || u.origin !== location.origin) {           // data: network first, last good copy offline
    e.respondWith(fetch(e.request).then(r => { if (r.ok) { const c = r.clone(); caches.open(RUNTIME).then(x => x.put(e.request, c)); } return r; })
      .catch(() => caches.match(e.request)));
  } else {                                                                         // shell: cache first
    e.respondWith(caches.match(e.request).then(m => m || fetch(e.request)));
  }
});
