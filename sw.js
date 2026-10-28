// rainy service worker: shows the rain alerts that the rainy server sends via Web Push.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("push", (e) => {
  let d = {};
  try { d = e.data.json(); } catch {}
  e.waitUntil(self.registration.showNotification(d.title || "rainy", {
    body: d.body || "", icon: "icon-192.png", badge: "icon-192.png",
    tag: d.tag || "rainy", renotify: true, data: d, // one notification per place, newer ones replace older ones
  }));
});

// Tapping the notification opens the app at that place.
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const d = e.notification.data || {};
  const url = new URL(d.lat != null ? `./?lat=${d.lat}&lon=${d.lon}` : "./", self.registration.scope).href;
  e.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
    const open = list.find((c) => "focus" in c);
    if (open) return open.navigate(url).then((c) => (c || open).focus());
    return self.clients.openWindow(url);
  }));
});
