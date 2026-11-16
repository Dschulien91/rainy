package app.schachcoach;

import android.Manifest;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.work.Constraints;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import org.json.JSONArray;
import org.json.JSONObject;

import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

/**
 * Every few hours (only with network and a battery that is not low) asks the public Chess.com API whether
 * there are games newer than the newest one in the app, and posts one notification if so. Nothing is analysed
 * in the background: that happens when the app is opened.
 */
public class NewGamesWorker extends Worker {
    static final String PREFS = "newgames";
    static final String WORK = "new-games";
    static final String CHANNEL = "new-games";

    public NewGamesWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    static void schedule(Context ctx, boolean enabled) {
        WorkManager wm = WorkManager.getInstance(ctx);
        if (!enabled) {
            wm.cancelUniqueWork(WORK);
            return;
        }
        Constraints c = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build();
        PeriodicWorkRequest req = new PeriodicWorkRequest.Builder(NewGamesWorker.class, 4, TimeUnit.HOURS)
                .setConstraints(c)
                .build();
        wm.enqueueUniquePeriodicWork(WORK, ExistingPeriodicWorkPolicy.KEEP, req);
    }

    private static String get(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        try {
            c.setRequestProperty("User-Agent", MainActivity.USER_AGENT);
            c.setConnectTimeout(15000);
            c.setReadTimeout(30000);
            if (c.getResponseCode() != 200) throw new Exception("HTTP " + c.getResponseCode());
            return new String(MainActivity.readAll(c.getInputStream()), StandardCharsets.UTF_8);
        } finally {
            c.disconnect();
        }
    }

    @NonNull
    @Override
    public Result doWork() {
        Context ctx = getApplicationContext();
        SharedPreferences p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        String user = p.getString("user", "");
        if (!p.getBoolean("enabled", false) || user.isEmpty()) return Result.success();
        long newest = Math.max(p.getLong("newest", 0), p.getLong("notified", 0));
        try {
            JSONArray archives = new JSONObject(get("https://api.chess.com/pub/player/"
                    + java.net.URLEncoder.encode(user.toLowerCase(), "UTF-8") + "/games/archives")).getJSONArray("archives");
            if (archives.length() == 0) return Result.success();
            JSONArray games = new JSONObject(get(archives.getString(archives.length() - 1))).getJSONArray("games");
            int count = 0;
            long latest = newest;
            String lastOpp = null;
            for (int i = 0; i < games.length(); i++) {
                JSONObject g = games.getJSONObject(i);
                long end = g.optLong("end_time", 0) * 1000L;
                if (end <= newest) continue;
                count++;
                if (end >= latest) {
                    latest = end;
                    JSONObject w = g.getJSONObject("white"), b = g.getJSONObject("black");
                    lastOpp = w.optString("username").equalsIgnoreCase(user) ? b.optString("username") : w.optString("username");
                }
            }
            if (count > 0) {
                p.edit().putLong("notified", latest).apply();
                notify(ctx, count, lastOpp);
            }
            return Result.success();
        } catch (Exception e) {
            return Result.retry();
        }
    }

    private static void notify(Context ctx, int count, String opponent) {
        if (Build.VERSION.SDK_INT >= 33
                && ctx.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return;
        NotificationManager nm = ctx.getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Neue Partien", NotificationManager.IMPORTANCE_DEFAULT));
        Intent open = new Intent(ctx, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(ctx, 0, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        String title = count == 1 ? "Neue Partie" + (opponent != null ? " gegen " + opponent : "") : count + " neue Partien";
        android.app.Notification n = new android.app.Notification.Builder(ctx, CHANNEL)
                .setSmallIcon(R.drawable.ic_notify)
                .setContentTitle(title)
                .setContentText("Tippe, damit Ole sie mit dir auswertet.")
                .setContentIntent(pi)
                .setAutoCancel(true)
                .build();
        nm.notify(1, n);
    }
}
