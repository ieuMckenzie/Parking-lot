Near-term (get it solid)

1. Integration tests with real images — You have test images and model weights already. A few tests that load the actual YOLO model, run detection on a
   known image, and verify OCR output would catch regressions when you change preprocessing or model configs. Mark them @pytest.mark.integration so they
   don't run on every save.

2. WebSocket or SSE for live detections — Right now the API only has REST polling. Your teammate's truck driver webapp would benefit from a
   /ws/detections endpoint that pushes new plate reads in real-time instead of requiring the frontend to poll /detections every second.

3. Database instead of CSV — The CSV logger works but doesn't scale. SQLite (or the D1 database you have on Cloudflare) would give you proper querying,
   pagination, and historical analytics. The CSVLogger interface is already clean enough to swap in a DBLogger behind it.

Medium-term (enable the webapps)

4. Parking spot assignment model — The current system detects plates but doesn't track where trucks are parked. Adding a spatial model (camera zones
   mapped to parking spots) would let the truck driver app show "you're assigned to spot B7" and the admin app show lot occupancy.

5. Admin dashboard data endpoints — Camera feed thumbnails, lot layout config, historical plate queries, detection statistics. Most of the plumbing is
   there via ScannerEngine — it's mostly new routes + a simple frontend.

6. Authentication on the API — Right now everything is wide open. Before deploying, add API keys or JWT auth, especially on the config/authorized CRUD
   endpoints.

Longer-term

7. Docker/deployment packaging — A Dockerfile that bundles the scanner + API, with model weights mounted as a volume. Makes deployment to the actual lot
   hardware reproducible.

8. Multi-lot support — Config-driven lot definitions so one deployment can manage multiple camera groups / parking areas.

9. WMS integration — The admin webapp connecting to a warehouse management system for dispatch coordination. This is mostly API glue once the data model
   exists.
