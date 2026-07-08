(ns bjs-drones.build-html
  "Generates docs/index.html with embedded Vega-Lite charts."
  (:require
   [bjs-drones.telemetry :as tel]
   [bjs-drones.pipeline :as pipe]
   [bjs-drones.baseline :as bl]
   [bjs-drones.hdc :as hdc]
   [bjs-drones.eval :as ev]
   [tablecloth.api :as tc]
   [clojure.data.json :as json])
  (:import [java.io File]))

;; --- Data ---

(defn build-data []
  (let [flights    (tel/generate-dataset :n-normal 10 :steps 800)
        all-windows (pipe/extract-features flights :window-size 20 :step 10)
        result     (ev/run-comparison all-windows :train-frac 0.7)
        hdc-model  (:hdc-model result)
        bl-model   (:bl-model result)
        test-ws    (:test-windows result)]
    {:flights flights :all-windows all-windows :result result
     :hdc-model hdc-model :bl-model bl-model :test-ws test-ws}))

;; --- Vega-Lite specs ---

(defn trajectory-spec [flights]
  (let [records (mapcat (fn [{:keys [flight-id records]}]
                          (map #(hash-map :lon (:lon %) :lat (:lat %)
                                          :anomaly (name (or (:anomaly %) :normal))
                                          :flight flight-id)
                               records))
                        flights)]
    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :title "Flight Trajectories — coloured by anomaly"
     :width 700 :height 400
     :data {:values records}
     :mark {:type "point" :size 3 :opacity 0.5}
     :encoding {:x {:field "lon" :type "quantitative" :title "Longitude"}
                :y {:field "lat" :type "quantitative" :title "Latitude"}
                :color {:field "anomaly" :type "nominal"}}}))

(defn score-timeline-spec [flights bl-model]
  (let [gps-flight (first (filter #(= (:flight-id %) "anomaly-gps-jam") flights))
        windows    (pipe/sliding-windows
                    (pipe/add-derived
                     (pipe/records->ds "anomaly-gps-jam" (:records gps-flight)))
                    {:window-size 20 :step 5})
        rows (map (fn [w]
                    {:t     (:window-start w)
                     :score (bl/anomaly-score bl-model w)
                     :anomaly (name (or (:anomaly-label w) :normal))})
                  windows)]
    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :title "Baseline Z-Score over Time — GPS-Jam Flight"
     :width 700 :height 280
     :data {:values rows}
     :layer [{:mark {:type "line" :color "steelblue"}
              :encoding {:x {:field "t" :type "quantitative" :title "Time (s)"}
                         :y {:field "score" :type "quantitative" :title "Anomaly Score"}}}
             {:mark {:type "rect" :opacity 0.15 :color "red"}
              :transform [{:filter "datum.anomaly !== 'normal'"}]
              :encoding {:x {:field "t" :type "quantitative"}
                         :x2 {:value 9999}}}]}))

(defn hdc-sim-spec [flights hdc-model]
  (let [gps-flight (first (filter #(= (:flight-id %) "anomaly-gps-jam") flights))
        windows    (pipe/sliding-windows
                    (pipe/add-derived
                     (pipe/records->ds "anomaly-gps-jam" (:records gps-flight)))
                    {:window-size 20 :step 5})
        rows (mapcat (fn [w]
                       (let [{:keys [scores]} (hdc/predict hdc-model w)]
                         (map (fn [[lbl sim]]
                                {:t (double (:window-start w))
                                 :label (name lbl)
                                 :sim (double sim)})
                              scores)))
                     windows)]
    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :title "HDC Cosine Similarity to Prototypes — GPS-Jam Flight"
     :width 700 :height 280
     :data {:values rows}
     :mark {:type "line"}
     :encoding {:x {:field "t" :type "quantitative" :title "Time (s)"}
                :y {:field "sim" :type "quantitative" :title "Cosine Similarity"}
                :color {:field "label" :type "nominal"}}}))

(defn scatter-spec [all-windows]
  (let [rows (map (fn [w]
                    {:label (name (:anomaly-label w))
                     :speed-mean (double (:speed-mean w 0))
                     :attitude (double (:attitude-magnitude-mean w 0))})
                  all-windows)]
    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :title "Speed vs Attitude — by Anomaly Class"
     :width 600 :height 380
     :data {:values rows}
     :mark {:type "point" :size 30 :opacity 0.6}
     :encoding {:x {:field "speed-mean" :type "quantitative" :title "Speed (mean)"}
                :y {:field "attitude" :type "quantitative" :title "Attitude Magnitude (mean)"}
                :color {:field "label" :type "nominal"}}}))

(defn confusion-spec [test-ws hdc-model]
  (let [preds  (map (fn [w]
                      {:true (name (:anomaly-label w))
                       :pred (name (:label (hdc/predict hdc-model w)))})
                    test-ws)
        classes (sort (distinct (mapcat (juxt :true :pred) preds)))
        rows    (for [t classes p classes]
                  {:true t :pred p
                   :count (count (filter #(and (= (:true %) t) (= (:pred %) p)) preds))})]
    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :title "HDC Confusion Matrix"
     :width 380 :height 380
     :data {:values rows}
     :layer [{:mark "rect"
              :encoding {:x {:field "pred" :type "nominal" :title "Predicted"}
                         :y {:field "true" :type "nominal" :title "True"}
                         :color {:field "count" :type "quantitative"
                                 :scale {:scheme "blues"}}}}
             {:mark {:type "text" :baseline "middle"}
              :encoding {:x {:field "pred" :type "nominal"}
                         :y {:field "true" :type "nominal"}
                         :text {:field "count" :type "quantitative"}
                         :color {:value "black"}}}]}))

;; --- HTML template ---

(defn chart-div [id spec]
  (str "<div id=\"" id "\"></div>\n"
       "<script>vegaEmbed('#" id "', " (json/write-str spec) ", {actions:false});</script>\n"))

(defn build-html [{:keys [flights all-windows hdc-model bl-model test-ws result]}]
  (let [bl   (:baseline result)
        hdc  (:hdc result)
        specs {:trajectories (trajectory-spec flights)
               :score-timeline (score-timeline-spec flights bl-model)
               :hdc-sims (hdc-sim-spec flights hdc-model)
               :scatter (scatter-spec all-windows)
               :confusion (confusion-spec test-ws hdc-model)}]
    (str "<!DOCTYPE html><html><head><meta charset='utf-8'>"
         "<title>bjs-drones-1 — UAV Anomaly Detection</title>"
         "<script src='https://cdn.jsdelivr.net/npm/vega@5'></script>"
         "<script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>"
         "<script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>"
         "<style>body{font-family:monospace;max-width:900px;margin:40px auto;padding:0 20px;background:#0d1117;color:#e6edf3}"
         "h1{color:#58a6ff}h2{color:#79c0ff;margin-top:2em}"
         ".metric{display:inline-block;margin:0 1em;background:#161b22;padding:.5em 1em;border-radius:6px;border:1px solid #30363d}"
         ".metric span{color:#3fb950;font-size:1.4em;display:block}"
         "</style></head><body>"
         "<h1>UAV Telemetry Anomaly Detection</h1>"
         "<p>Two classifiers: z-score baseline (binary) vs HDC multi-class (5 labels, pure Clojure, no ML framework).</p>"

         "<h2>Results</h2>"
         "<div class='metric'>Baseline F1<span>" (format "%.3f" (double (:f1 bl))) "</span></div>"
         "<div class='metric'>Baseline Accuracy<span>" (format "%.1f%%" (* 100 (double (:accuracy bl)))) "</span></div>"
         "<div class='metric'>HDC Accuracy<span>" (format "%.1f%%" (* 100 (double (:accuracy hdc)))) "</span></div>"
         "<div class='metric'>GPS-Jam Detection<span>"
         (format "%.0f%%" (* 100 (double (get-in hdc [:per-class :gps-jam :accuracy] 0)))) "</span></div>"

         "<h2>Flight Trajectories</h2>"
         (chart-div "trajectories" (:trajectories specs))

         "<h2>Baseline Z-Score over Time (GPS-Jam flight)</h2>"
         (chart-div "score-timeline" (:score-timeline specs))

         "<h2>HDC: Cosine Similarity to Prototypes (GPS-Jam flight)</h2>"
         (chart-div "hdc-sims" (:hdc-sims specs))

         "<h2>Feature Space — Speed vs Attitude</h2>"
         (chart-div "scatter" (:scatter specs))

         "<h2>HDC Confusion Matrix</h2>"
         (chart-div "confusion" (:confusion specs))

         "</body></html>")))

(defn -main [& _]
  (.mkdirs (File. "docs"))
  (println "Building data...")
  (let [data (build-data)
        html (build-html data)]
    (spit "docs/index.html" html)
    (println "Wrote docs/index.html")
    (ev/print-report (:result data))))
