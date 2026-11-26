// Offline support: saves the page, the ONNX runtime and the model on first visit.
// Bump VERSION after uploading a new model to Hugging Face, or phones keep using the saved one.
const VERSION = "v1";
const MODEL_REPO = "https://huggingface.co/laimengeung/khmer-lstm-nextword/resolve/main/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const PRECACHE = ["./", "manifest.json", "icon-192.png", "icon-512.png",
  ORT + "ort.min.js", ORT + "ort-wasm-simd-threaded.mjs", ORT + "ort-wasm-simd-threaded.wasm",
  MODEL_REPO + "model.onnx", MODEL_REPO + "vocab.json"];

self.addEventListener("install", e => e.waitUntil(
  caches.open(VERSION).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting())));

self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
    .then(() => self.clients.claim())));

const save = (req, res) => {
  if (res.ok) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
  return res;
};

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const sameSite = new URL(req.url).origin === location.origin;
  e.respondWith(sameSite
    // page files: try the network so pushed updates show up, fall back to the saved copy offline
    ? fetch(req).then(res => save(req, res)).catch(() => caches.match(req, { ignoreSearch: true }))
    // model, runtime, fonts: big and versioned, so use the saved copy first
    : caches.match(req).then(hit => hit || fetch(req).then(res => save(req, res))));
});
