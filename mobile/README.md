# EdgeAgent on your phone

The phone app is a thin client: the engine (data fetching, models, Brainbox) runs on a PC or server, and the phone shows its results.

## Fastest: install as an app (no store, no build)
1. Run the engine somewhere reachable: `edgeagent serve --host 0.0.0.0` (or the Windows app / docker compose).
2. **Protect it**: set `EDGEAGENT_API_TOKEN` to a long random string, and put it behind HTTPS (Caddy, nginx, or a Cloudflare Tunnel). Never expose it open to the internet.
3. On the phone open `https://your-host/app/`, then Settings -> enter the token. Android Chrome: menu -> Add to Home screen. iPhone Safari: Share -> Add to Home Screen.

## Optional: Android APK (debug build)
Actions -> `android-apk` -> Run workflow -> download `app-debug.apk` and sideload it. In the app's Settings set the API address (must be `https://`) and token. This workflow has not been run by the author: expect to fix small Gradle/Capacitor issues on the first run.
