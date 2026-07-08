(ns bjs-drones.baseline
  "Z-score one-class anomaly detector.
   Trained on normal windows only; anomaly score = max z-score across features.")

(def feature-keys
  [:speed-mean :speed-std :speed-max
   :alt-mean :alt-std
   :attitude-magnitude-mean :attitude-magnitude-std :attitude-magnitude-max
   :battery-mean :battery-std
   :gps-fix-mean :gps-fix-min])

(defn- col-stats [windows k]
  (let [vals (mapv #(double (get % k 0.0)) windows)
        n    (count vals)
        mu   (/ (reduce + vals) n)
        sig  (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) vals)) n))]
    {:mean mu :std (max sig 1e-9)}))

(defn fit
  "Returns model trained on normal windows."
  [windows]
  {:stats (into {} (for [k feature-keys]
                     [k (col-stats windows k)]))})

(defn anomaly-score
  "Max z-score across features for a single window."
  [model window]
  (let [stats (:stats model)]
    (reduce max
            (for [k feature-keys
                  :let [{:keys [mean std]} (get stats k)
                        v (double (get window k 0.0))]]
              (Math/abs (/ (- v mean) std))))))

(defn predict
  "Returns {:score num :anomaly? bool :label kw}."
  [model window & {:keys [threshold] :or {threshold 3.5}}]
  (let [score (anomaly-score model window)]
    {:score    score
     :anomaly? (> score threshold)
     :label    (if (> score threshold) :anomaly :normal)}))

(defn evaluate
  "Accuracy, precision, recall over labeled windows."
  [model windows & {:keys [threshold] :or {threshold 3.5}}]
  (let [results (map #(assoc (predict model % :threshold threshold)
                             :true-label (:anomaly-label %))
                     windows)
        true-pos  (count (filter #(and (:anomaly? %) (not= (:true-label %) :normal)) results))
        false-pos (count (filter #(and (:anomaly? %) (= (:true-label %) :normal)) results))
        false-neg (count (filter #(and (not (:anomaly? %)) (not= (:true-label %) :normal)) results))
        true-neg  (count (filter #(and (not (:anomaly? %)) (= (:true-label %) :normal)) results))
        total     (count results)
        prec      (if (zero? (+ true-pos false-pos)) 0.0
                      (/ true-pos (+ true-pos false-pos)))
        rec       (if (zero? (+ true-pos false-neg)) 0.0
                      (/ true-pos (+ true-pos false-neg)))]
    {:accuracy  (/ (+ true-pos true-neg) total)
     :precision prec
     :recall    rec
     :f1        (if (zero? (+ prec rec)) 0.0
                    (* 2 (/ (* prec rec) (+ prec rec))))
     :tp true-pos :fp false-pos :fn false-neg :tn true-neg}))
