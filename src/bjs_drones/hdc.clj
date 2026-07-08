(ns bjs-drones.hdc
  "Hyperdimensional Computing classifier for UAV telemetry anomaly detection.

   Architecture:
   - Bipolar hypervectors in R^D, D=10000
   - Level coding: each scalar feature → gradient HV (similar values → similar HVs)
   - Window encoding: superposition of feature HVs
   - Prototype learning: bundle all window HVs per class
   - Classification: cosine similarity to prototypes"
  (:require [bjs-drones.baseline :as bl]))

(def D 10000)

(def all-labels [:normal :gps-jam :motor-failure :battery-drain :hijack])

;; --- Hypervector primitives ---

(defn seed-hv
  "Random bipolar hypervector {-1.0, +1.0}^D."
  [^long seed]
  (let [rng (java.util.Random. seed)
        arr (double-array D)]
    (dotimes [i D]
      (aset arr i (if (.nextBoolean rng) 1.0 -1.0)))
    arr))

(defn bundle
  "Superposition (elementwise add) of multiple HVs → double[] accumulator.
   Call binarize to get a proper HV."
  [hvs]
  (let [acc (double-array D)]
    (doseq [^doubles hv hvs]
      (dotimes [i D]
        (aset acc i (+ (aget acc i) (aget hv i)))))
    acc))

(defn binarize
  "Sign function: positive → 1.0, negative → -1.0, zero → 1.0."
  [^doubles acc]
  (let [arr (double-array D)]
    (dotimes [i D]
      (aset arr i (if (>= (aget acc i) 0.0) 1.0 -1.0)))
    arr))

(defn cosine-sim
  "Cosine similarity between two bipolar HVs."
  [^doubles a ^doubles b]
  (let [dot  (double-array 1)
        ma   (double-array 1)
        mb   (double-array 1)]
    (dotimes [i D]
      (aset dot 0 (+ (aget dot 0) (* (aget a i) (aget b i))))
      (aset ma  0 (+ (aget ma  0) (* (aget a i) (aget a i))))
      (aset mb  0 (+ (aget mb  0) (* (aget b i) (aget b i)))))
    (/ (aget dot 0) (* (Math/sqrt (aget ma 0)) (Math/sqrt (aget mb 0))))))

;; --- Level coding ---

(defn level-hv
  "Encodes a scalar value v ∈ [v-min, v-max] as a bipolar HV.
   Uses interpolation between two seed vectors: similar values → similar HVs.
   fraction=0 → base-hv, fraction=1 → complement of base-hv."
  [^doubles base-hv fraction]
  (let [fraction (max 0.0 (min 1.0 (double fraction)))
        n-flip   (long (* fraction (/ D 2)))
        arr      (aclone base-hv)]
    ;; Flip the first n-flip components deterministically
    (dotimes [i n-flip]
      (aset arr i (* -1.0 (aget arr i))))
    arr))

;; --- Codec ---

(defn build-codec
  "Builds the feature codec from training windows.
   For each feature: a seed HV + min/max range for level coding."
  [train-windows]
  (into {}
        (map-indexed
         (fn [idx k]
           (let [vals  (keep #(get % k) train-windows)
                 v-min (if (seq vals) (reduce min vals) 0.0)
                 v-max (if (seq vals) (reduce max vals) 1.0)
                 range (max (- v-max v-min) 1e-9)]
             [k {:base  (seed-hv (+ (* idx 1000) 42))
                 :v-min v-min
                 :v-max v-max
                 :range range}]))
         bl/feature-keys)))

(defn encode-window
  "Encodes a feature map → bipolar HV via superposition of level HVs."
  [codec window]
  (->> bl/feature-keys
       (mapv (fn [k]
               (let [{:keys [base v-min range]} (get codec k)
                     v (double (get window k 0.0))
                     fraction (/ (- v v-min) range)]
                 (level-hv base fraction))))
       bundle
       binarize))

;; --- Training ---

(defn train
  "Returns model: {:codec ... :prototypes {label → HV}}."
  [train-windows]
  (let [codec (build-codec train-windows)
        by-label (group-by :anomaly-label train-windows)
        prototypes (into {}
                         (for [[label windows] by-label
                               :when (seq windows)]
                           [label
                            (-> (mapv #(encode-window codec %) windows)
                                bundle
                                binarize)]))]
    {:codec      codec
     :prototypes prototypes}))

;; --- Inference ---

(defn predict
  "Returns {:label kw :scores {label → sim} :confidence float}."
  [model window]
  (let [{:keys [codec prototypes]} model
        hv     (encode-window codec window)
        scores (into {}
                     (for [[label proto] prototypes]
                       [label (cosine-sim hv proto)]))
        best   (apply max-key val scores)]
    {:label      (key best)
     :scores     scores
     :confidence (val best)}))

;; --- Evaluation ---

(defn evaluate
  "Per-class and overall accuracy."
  [model windows]
  (let [results (map (fn [w]
                       (assoc (predict model w)
                              :true-label (:anomaly-label w)))
                     windows)
        correct (count (filter #(= (:label %) (:true-label %)) results))
        total   (count results)
        by-class (group-by :true-label results)]
    {:accuracy   (/ correct total)
     :n-correct  correct
     :n-total    total
     :per-class  (into {}
                       (for [[lbl rs] by-class]
                         [lbl {:accuracy (/ (count (filter #(= (:label %) (:true-label %)) rs))
                                            (count rs))
                               :n        (count rs)}]))}))
