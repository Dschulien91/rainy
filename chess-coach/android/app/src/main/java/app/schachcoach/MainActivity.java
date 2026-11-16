package app.schachcoach;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.WindowInsets;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

import androidx.webkit.WebViewAssetLoader;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * A thin shell around the web app in assets/ (web/ in the repository):
 * - serves the assets from https://appassets.androidplatform.net so the Stockfish worker and WASM load normally,
 * - forwards /proxy?u=https://api.chess.com/... to Chess.com (no CORS limits inside the app),
 * - text-to-speech for the coach, battery state, "share" from the Chess.com app, background check for new games.
 */
public class MainActivity extends Activity {
    static final String ORIGIN = "https://appassets.androidplatform.net";
    static final String USER_AGENT = "SchachCoach-Android/1.0";

    private WebView web;
    private WebViewAssetLoader assets;
    private TextToSpeech tts;
    private boolean ttsReady = false;
    private String pendingSpeech = null;
    private String sharedText = null;
    private boolean pageLoaded = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.parseColor("#2a2926"));
        getWindow().setNavigationBarColor(Color.parseColor("#1f1e1b"));

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.parseColor("#1f1e1b"));
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#1f1e1b"));
        root.addView(web, new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        if (Build.VERSION.SDK_INT >= 30) {
            // Android 15 draws apps edge to edge: keep the page clear of the status and navigation bars.
            root.setOnApplyWindowInsetsListener((v, insets) -> {
                android.graphics.Insets i = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout() | WindowInsets.Type.ime());
                v.setPadding(i.left, i.top, i.right, i.bottom);
                return WindowInsets.CONSUMED;
            });
        }
        setContentView(root);

        assets = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setMediaPlaybackRequiresUserGesture(true);
        s.setTextZoom(100);

        web.addJavascriptInterface(new Bridge(), "AndroidBridge");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if ("appassets.androidplatform.net".equals(u.getHost()) && "/proxy".equals(u.getPath())) {
                    return proxy(u.getQueryParameter("u"));
                }
                WebResourceResponse r = assets.shouldInterceptRequest(u);
                if (r != null) fixMime(r, u.getPath());
                return r;
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if ("appassets.androidplatform.net".equals(u.getHost())) return false;
                openExternal(u.toString());
                return true;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                pageLoaded = true;
                if (sharedText != null) web.evaluateJavascript("window.__checkShared && window.__checkShared()", null);
            }
        });

        if (Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                    android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::handleBack);
        }

        readShared(getIntent());
        web.loadUrl(ORIGIN + "/assets/index.html");
    }

    private static void fixMime(WebResourceResponse r, String path) {
        if (path == null) return;
        if (path.endsWith(".wasm")) r.setMimeType("application/wasm");
        else if (path.endsWith(".js")) r.setMimeType("text/javascript");
        else if (path.endsWith(".css")) r.setMimeType("text/css");
        else if (path.endsWith(".svg")) r.setMimeType("image/svg+xml");
        else if (path.endsWith(".json")) r.setMimeType("application/json");
        else if (path.endsWith(".html")) r.setMimeType("text/html");
    }

    /** Fetches a Chess.com API URL for the page (only https://api.chess.com is allowed). Runs on a WebView IO thread. */
    private WebResourceResponse proxy(String target) {
        Map<String, String> headers = new HashMap<>();
        headers.put("Cache-Control", "no-store");
        try {
            Uri t = Uri.parse(target == null ? "" : target);
            if (!"https".equals(t.getScheme()) || !"api.chess.com".equals(t.getHost())) {
                return new WebResourceResponse("text/plain", "utf-8", 403, "Forbidden", headers, new ByteArrayInputStream(new byte[0]));
            }
            byte[] body;
            int code;
            HttpURLConnection c = (HttpURLConnection) new URL(target).openConnection();
            try {
                c.setRequestProperty("User-Agent", USER_AGENT);
                c.setRequestProperty("Accept", "application/json");
                c.setConnectTimeout(15000);
                c.setReadTimeout(30000);
                code = c.getResponseCode();
                InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
                body = in == null ? new byte[0] : readAll(in);
            } finally {
                c.disconnect();
            }
            return new WebResourceResponse("application/json", "utf-8", code, code >= 400 ? "Error" : "OK", headers, new ByteArrayInputStream(body));
        } catch (Exception e) {
            return new WebResourceResponse("text/plain", "utf-8", 502, "Bad Gateway", headers,
                    new ByteArrayInputStream(String.valueOf(e.getMessage()).getBytes(StandardCharsets.UTF_8)));
        }
    }

    static byte[] readAll(InputStream in) throws java.io.IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[16384];
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        in.close();
        return out.toByteArray();
    }

    private void readShared(Intent intent) {
        if (intent == null) return;
        String text = null;
        if (Intent.ACTION_SEND.equals(intent.getAction())) {
            CharSequence cs = intent.getCharSequenceExtra(Intent.EXTRA_TEXT);
            if (cs != null) text = cs.toString();
            if (text == null) {
                Uri stream = intent.getParcelableExtra(Intent.EXTRA_STREAM);
                if (stream != null) text = readUri(stream);
            }
        } else if (Intent.ACTION_VIEW.equals(intent.getAction()) && intent.getData() != null) {
            text = readUri(intent.getData());
        }
        if (text != null && !text.trim().isEmpty()) sharedText = text;
    }

    private String readUri(Uri uri) {
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            if (in == null) return null;
            return new String(readAll(in), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        readShared(intent);
        if (pageLoaded && sharedText != null) web.evaluateJavascript("window.__checkShared && window.__checkShared()", null);
    }

    private void handleBack() {
        web.evaluateJavascript("(window.__onBack && window.__onBack()) ? 1 : 0", value -> {
            if (!"1".equals(value)) {
                moveTaskToBack(true);
            }
        });
    }

    @SuppressWarnings("deprecation")
    @Override
    public void onBackPressed() {
        handleBack();
    }

    @Override
    protected void onResume() {
        super.onResume();
        web.onResume();
    }

    @Override
    protected void onPause() {
        web.onPause();
        if (tts != null) tts.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (tts != null) tts.shutdown();
        web.destroy();
        super.onDestroy();
    }

    private void openExternal(String url) {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
        } catch (Exception ignored) {
        }
    }

    private void speakNow(String text) {
        if (tts == null) {
            pendingSpeech = text;
            tts = new TextToSpeech(this, status -> {
                if (status == TextToSpeech.SUCCESS) {
                    tts.setLanguage(Locale.GERMAN);
                    ttsReady = true;
                    if (pendingSpeech != null) tts.speak(pendingSpeech, TextToSpeech.QUEUE_FLUSH, null, "coach");
                    pendingSpeech = null;
                }
            });
        } else if (ttsReady) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "coach");
        } else {
            pendingSpeech = text;
        }
    }

    /** Methods the web app can call as window.AndroidBridge.*. They run on a background thread. */
    class Bridge {
        @JavascriptInterface
        public void speak(String text) {
            runOnUiThread(() -> speakNow(text));
        }

        @JavascriptInterface
        public void stopSpeaking() {
            runOnUiThread(() -> {
                pendingSpeech = null;
                if (tts != null) tts.stop();
            });
        }

        @JavascriptInterface
        public boolean isCharging() {
            Intent b = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
            if (b == null) return false;
            int st = b.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
            return st == BatteryManager.BATTERY_STATUS_CHARGING || st == BatteryManager.BATTERY_STATUS_FULL;
        }

        @JavascriptInterface
        public String takeSharedText() {
            String t = sharedText;
            sharedText = null;
            return t;
        }

        @JavascriptInterface
        public void openUrl(String url) {
            if (url != null && (url.startsWith("https://www.chess.com/") || url.startsWith("https://chess.com/"))) {
                runOnUiThread(() -> openExternal(url));
            }
        }

        /** Username + newest known game, used by the background check. enabled=false stops it. */
        @JavascriptInterface
        public void setSyncState(String user, double newestEndTimeMs, boolean enabled) {
            SharedPreferences p = getSharedPreferences(NewGamesWorker.PREFS, MODE_PRIVATE);
            p.edit().putString("user", user == null ? "" : user)
                    .putLong("newest", Math.max(p.getLong("newest", 0), (long) newestEndTimeMs))
                    .putBoolean("enabled", enabled).apply();
            if (enabled && Build.VERSION.SDK_INT >= 33
                    && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                runOnUiThread(() -> requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1));
            }
            NewGamesWorker.schedule(getApplicationContext(), enabled);
        }
    }
}
